"""迷你 DeepSeek 风格 LLM 推理引擎（教学版）。

目标：用几百行纯 PyTorch 把 DeepSeek 系模型的三个核心机制写清楚——
  1. MLA（Multi-head Latent Attention）：KV cache 只存低秩压缩向量
  2. MoE：门控路由 + top-k 专家 + 常开共享专家
  3. 支持增量 KV cache 的生成循环（投机解码见 generate.py）

与 DeepInfer（生产版）的对应关系在注释里以 [DeepInfer] 标注。
刻意简化掉的东西：FP8 量化、张量/专家并行、CUDA graph、
YoCo KV 共享、稀疏索引注意力（MQAIndexer）、Engram、HC 头。

只依赖 torch，CPU 可跑。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MiniConfig:
    """模型尺寸。默认配置小到 CPU 秒级前向，但结构是完整的。"""

    vocab_size: int = 1024
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 4
    q_lora_rank: int = 32          # Q 的低秩压缩维
    kv_lora_rank: int = 32         # KV 的低秩压缩维（cache 存的就是它 + rope key）
    qk_nope_head_dim: int = 16     # 每 head 非位置维
    qk_rope_head_dim: int = 16     # RoPE 维（所有 head 共享一条 key）
    v_head_dim: int = 32
    moe_intermediate_size: int = 96
    n_routed_experts: int = 8
    num_experts_per_tok: int = 2
    n_shared_experts: int = 1
    routed_scaling_factor: float = 1.0
    max_seq_len: int = 512
    rope_theta: float = 10000.0

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


class RMSNorm(nn.Module):
    """RMSNorm（无均值中心化）。DeepSeek 系全部用它。"""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def rope_cos_sin(seq_len: int, head_dim: int, theta: float, device, dtype):
    """预计算 RoPE 的 cos/sin 表。对 rotate-half 约定：一半维度旋转。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                      # [T, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)               # [T, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, D]（或 [B, T, D]，会自动升维），只对最后一维做旋转。"""
    squeeze_h = x.dim() == 3
    if squeeze_h:
        x = x.unsqueeze(1)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    out = x * cos[None, None] + rotated * sin[None, None]
    return out.squeeze(1) if squeeze_h else out


class MLA(nn.Module):
    """多头潜在注意力（教学版，显式升投影写法）。

    DeepSeek-V2/V3 论文的记号：
        c_Q = W_DQ h            （q 的低秩压缩）
        c_KV = W_DKV h          （KV 的低秩压缩 —— cache 里存它）
        k_rope = RoPE(W_KR h)   （所有 head 共享的位置 key —— cache 里存它）
        q_nope_i = W_UQ_i c_Q,  k_nope_i = W_UK_i c_KV,  v_i = W_UV_i c_KV

    KV cache 每 token 只需 kv_lora_rank + qk_rope_head_dim 个数，
    而 MHA 需要 num_heads × (qk_head_dim + v_head_dim) 个——这就是 MLA 省显存的原理。

    [DeepInfer] 生产版用"权重吸收"把 W_UK/W_UV 折进注意力里避免每步升投影，
    还有 FP8 量化、KV Gate 压缩（attn.py 的 Compressor）。教学版每步对整段
    cache 做升投影，O(T) 矩阵乘换来一眼看懂的直白实现。
    """

    def __init__(self, cfg: MiniConfig) -> None:
        super().__init__()
        self.cfg = cfg
        H = cfg.num_heads

        # Q 路径：降维 → norm → 升维(nope) + RoPE 分支
        self.w_dq = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.w_uq_nope = nn.Linear(cfg.q_lora_rank, H * cfg.qk_nope_head_dim, bias=False)
        self.w_qr = nn.Linear(cfg.q_lora_rank, cfg.qk_rope_head_dim, bias=False)

        # KV 路径：压缩量 c_KV + 共享 rope key
        self.w_dkv = nn.Linear(cfg.hidden_size, cfg.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.w_kr = nn.Linear(cfg.hidden_size, cfg.qk_rope_head_dim, bias=False)
        # 升投影（每 head 独立，实现上一次 Linear 出全部 head 再 reshape）
        self.w_ukc = nn.Linear(cfg.kv_lora_rank, H * cfg.qk_nope_head_dim, bias=False)
        self.w_uvc = nn.Linear(cfg.kv_lora_rank, H * cfg.v_head_dim, bias=False)

        self.w_o = nn.Linear(H * cfg.v_head_dim, cfg.hidden_size, bias=False)

    def forward(
        self,
        h: torch.Tensor,                       # [B, Tq, hidden]
        c_kv_cache: torch.Tensor | None,       # [B, Tk, kv_lora_rank] 或 None
        k_rope_cache: torch.Tensor | None,     # [B, Tk, rope] 或 None
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (输出, 新 c_KV, 新 k_rope)；调用方负责拼进 cache。"""
        cfg = self.cfg
        B, Tq, _ = h.shape
        Tk = Tq if c_kv_cache is None else c_kv_cache.shape[1] + Tq  # 拼接后的总长度

        # --- Q ---
        c_q = self.q_norm(self.w_dq(h))                          # [B, Tq, q_lora]
        q_nope = self.w_uq_nope(c_q).view(B, Tq, cfg.num_heads, cfg.qk_nope_head_dim).transpose(1, 2)
        q_rope = apply_rope(self.w_qr(c_q), cos[Tk - Tq: Tk], sin[Tk - Tq: Tk])  # [B, Tq, rope]

        # --- KV（本 token 新产生的压缩量） ---
        c_kv_new = self.kv_norm(self.w_dkv(h))                   # [B, Tq, kv_lora]
        k_rope_new = apply_rope(self.w_kr(h), cos[Tk - Tq: Tk], sin[Tk - Tq: Tk])  # [B, Tq, rope]

        # cache 里是历史压缩量；本步新算的和历史拼起来
        c_kv = c_kv_new if c_kv_cache is None else torch.cat([c_kv_cache, c_kv_new], dim=1)
        k_rope = k_rope_new if k_rope_cache is None else torch.cat([k_rope_cache, k_rope_new], dim=1)

        # --- 升投影回每 head 的 k/v ---
        k_nope = self.w_ukc(c_kv).view(B, Tk, cfg.num_heads, cfg.qk_nope_head_dim).transpose(1, 2)
        v = self.w_uvc(c_kv).view(B, Tk, cfg.num_heads, cfg.v_head_dim).transpose(1, 2)
        k_rope_h = k_rope.unsqueeze(1).expand(B, cfg.num_heads, Tk, cfg.qk_rope_head_dim)
        q_rope_h = q_rope.unsqueeze(1).expand(B, cfg.num_heads, Tq, cfg.qk_rope_head_dim)

        # --- 注意力打分：非位置部分 + 位置部分 ---
        scale = 1.0 / math.sqrt(cfg.qk_head_dim)
        scores = (q_nope @ k_nope.transpose(-1, -2) + q_rope_h @ k_rope_h.transpose(-1, -2)) * scale

        if Tq == Tk:  # prefill：因果掩码
            mask = torch.ones(Tq, Tk, dtype=torch.bool, device=h.device).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
        # decode（Tq=1, Tk=全历史）天然只看过去，无需掩码

        attn = F.softmax(scores.float(), dim=-1).type_as(v)
        out = (attn @ v).transpose(1, 2).reshape(B, Tq, cfg.num_heads * cfg.v_head_dim)
        return self.w_o(out), c_kv, k_rope


class ExpertMLP(nn.Module):
    """SwiGLU 专家。生产版是 FP8 + DeepGEMM（[DeepInfer] moe.py: use_fp8_gemm）。"""

    def __init__(self, cfg: MiniConfig, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepseekMoE(nn.Module):
    """Top-k 路由 MoE + 常开共享专家。

    [DeepInfer] 生产版（v4_modules/moe.py TopKGate）：
      - 打分 sqrtsoftplus + noaux_tc top-k + FP32 负载修正 bias（auxiliary-loss-free）
      - 文本/图像 token 两套 bias
      - 冗余专家（EPLB 迁移热门专家）
    教学版用 softmax + top-k + 权重归一化，机制一致、细节从简。
    """

    def __init__(self, cfg: MiniConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gate = nn.Linear(cfg.hidden_size, cfg.n_routed_experts, bias=False)
        self.experts = nn.ModuleList(
            ExpertMLP(cfg, cfg.moe_intermediate_size) for _ in range(cfg.n_routed_experts)
        )
        self.shared_experts = nn.ModuleList(
            ExpertMLP(cfg, cfg.moe_intermediate_size) for _ in range(cfg.n_shared_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = F.softmax(self.gate(x.float()), dim=-1)          # [B, T, E]
        topk_w, topk_i = torch.topk(scores, self.cfg.num_experts_per_tok, dim=-1)
        topk_w = topk_w / topk_w.sum(-1, keepdim=True)            # 归一化（norm_topk_prob）

        out = torch.zeros_like(x)
        flat_x = x.reshape(-1, x.size(-1))
        flat_w = topk_w.reshape(-1, self.cfg.num_experts_per_tok)
        flat_i = topk_i.reshape(-1, self.cfg.num_experts_per_tok)
        for e in range(self.cfg.n_routed_experts):
            mask = flat_i == e                                    # [N, topk]
            if not mask.any():
                continue
            rows = mask.any(-1)                                   # 命中专家 e 的 token
            expert_out = self.experts[e](flat_x[rows])            # [n, hidden]
            w = (flat_w[rows] * mask[rows]).sum(-1, keepdim=True)  # 该 token 对 e 的权重
            out.reshape(-1, x.size(-1))[rows] += expert_out * w

        for shared in self.shared_experts:                        # 共享专家常开
            out = out + shared(x)
        return out * self.cfg.routed_scaling_factor


class TransformerBlock(nn.Module):
    """pre-norm：h = h + Attn(norm(h)); h = h + MoE(norm(h))"""

    def __init__(self, cfg: MiniConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size)
        self.attn = MLA(cfg)
        self.moe_norm = RMSNorm(cfg.hidden_size)
        self.moe = DeepseekMoE(cfg)

    def forward(self, h, c_kv, k_rope, cos, sin):
        a, c_kv, k_rope = self.attn(self.attn_norm(h), c_kv, k_rope, cos, sin)
        h = h + a
        h = h + self.moe(self.moe_norm(h))
        return h, c_kv, k_rope


class DeepSeekMini(nn.Module):
    """整体模型。forward 同时支持 prefill（整段）与 decode（单 token + cache）。"""

    def __init__(self, cfg: MiniConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MiniConfig()
        self.embed = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_size)
        self.blocks = nn.ModuleList(TransformerBlock(self.cfg) for _ in range(self.cfg.num_layers))
        self.norm = RMSNorm(self.cfg.hidden_size)
        self.lm_head = nn.Linear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False)
        self._init_weights()
        cos, sin = rope_cos_sin(self.cfg.max_seq_len, self.cfg.qk_rope_head_dim,
                                self.cfg.rope_theta, "cpu", torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,                  # [B, T]（decode 时 T=1）
        c_kv_cache: list[torch.Tensor] | None = None,   # 每层一份 [B, T_hist, kv_lora]
        k_rope_cache: list[torch.Tensor] | None = None,  # 每层一份 [B, T_hist, rope]
    ):
        """返回 (logits, 新 c_kv 列表, 新 k_rope 列表)。cache 传 None 即 prefill。"""
        h = self.embed(tokens)
        new_c_kv, new_k_rope = [], []
        for i, block in enumerate(self.blocks):
            c_prev = c_kv_cache[i] if c_kv_cache is not None else None
            r_prev = k_rope_cache[i] if k_rope_cache is not None else None
            h, c_i, r_i = block(h, c_prev, r_prev, self.rope_cos, self.rope_sin)
            new_c_kv.append(c_i)
            new_k_rope.append(r_i)
        logits = self.lm_head(self.norm(h))
        return logits, new_c_kv, new_k_rope

    @torch.no_grad()
    def step(self, token: int, c_kv_cache: list, k_rope_cache: list) -> torch.Tensor:
        """增量 decode：输入单个 token id，返回下一 token 的 logits。"""
        x = torch.tensor([[token]])
        logits, c_kv_cache[:], k_rope_cache[:] = self.forward(x, c_kv_cache, k_rope_cache)
        return logits[0, -1]


def mla_cache_tokens_per_token(cfg: MiniConfig) -> int:
    """MLA 每层每 token 的 cache 元素数：kv_lora_rank + rope 维。"""
    return cfg.kv_lora_rank + cfg.qk_rope_head_dim


def mha_cache_tokens_per_token(cfg: MiniConfig) -> int:
    """等尺寸 MHA 每层每 token 的 cache 元素数：heads × (k_dim + v_dim)。"""
    return cfg.num_heads * (cfg.qk_head_dim + cfg.v_head_dim)
