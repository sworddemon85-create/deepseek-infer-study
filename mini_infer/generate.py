"""生成循环：KV cache 增量解码、采样、投机解码（draft-verify）。

[DeepInfer] 对应关系：
  - KVCache          ↔ csrc/engine/block_manager.hpp（生产版是显存块管理 + pin/unpin）
  - 采样             ↔ layers/sampler.py（生产版含 top-k/top-p/min-p、结构化输出、
                        MTP 验证走 fused kernel）
  - speculative_generate ↔ DFlash + 置信度调度（mtp/dflash.py）：
                        生产版由 ConfidenceHead 预测每个 draft 的接受概率、
                        按置信度决定验证长度；教学版固定 draft 长度 + 经典
                        接受/拒绝规则。

运行：python mini_infer/generate.py
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DeepSeekMini, MiniConfig  # noqa: E402


class KVCache:
    """每层两个张量：c_kv [1, T, kv_lora] 与 k_rope [1, T, rope]。

    这就是 MLA 的卖点——对照 model.py 里 mla_cache_tokens_per_token /
    mha_cache_tokens_per_token 算一算省了多少。
    crop() 支持投机解码验证失败时的回滚（生产版 BlockManager 同样要回滚）。
    """

    def __init__(self) -> None:
        self.c_kv: list[torch.Tensor] = []
        self.k_rope: list[torch.Tensor] = []

    @property
    def length(self) -> int:
        return self.c_kv[0].shape[1] if self.c_kv else 0

    def crop(self, new_len: int) -> None:
        for i in range(len(self.c_kv)):
            self.c_kv[i] = self.c_kv[i][:, :new_len]
            self.k_rope[i] = self.k_rope[i][:, :new_len]


@torch.no_grad()
def prefill(model: DeepSeekMini, tokens: list[int], cache: KVCache) -> torch.Tensor:
    """整段预填，返回最后一个位置的对数。"""
    x = torch.tensor([tokens])
    logits, c_kv, k_rope = model.forward(x)
    cache.c_kv = [t for t in c_kv]
    cache.k_rope = [t for t in k_rope]
    return logits[0, -1]


@torch.no_grad()
def decode_step(model: DeepSeekMini, token: int, cache: KVCache) -> torch.Tensor:
    """增量解码一步。"""
    x = torch.tensor([[token]])
    logits, c_kv, k_rope = model.forward(x, cache.c_kv, cache.k_rope)
    cache.c_kv = c_kv
    cache.k_rope = k_rope
    return logits[0, -1]


def sample_token(logits: torch.Tensor, temperature: float = 0.0, top_k: int = 0,
                 top_p: float = 1.0, generator: torch.Generator | None = None) -> int:
    """temperature=0 即贪心；否则 temperature/top-k/top-p 采样。

    [DeepInfer] 结构化输出在这一步之前对 logits 加词表掩码
    （layers/tilelang_ops/structure_output.py），教学版略。
    """
    if temperature <= 0:
        return int(logits.argmax())
    probs = F.softmax(logits / temperature, dim=-1)
    if top_k > 0:
        probs, idx = torch.topk(probs, top_k)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = cum - sorted_probs < top_p                    # 至少保留第一个
        probs = sorted_probs[keep] / sorted_probs[keep].sum()
        idx = idx[keep] if top_k > 0 else sorted_idx[keep]
    choice = torch.multinomial(probs, 1, generator=generator)
    return int(idx[choice])


def generate(model: DeepSeekMini, prompt: list[int], max_new_tokens: int,
             temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0,
             seed: int | None = None) -> list[int]:
    """普通自回归生成（每步 1 次前向）。"""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    cache = KVCache()
    logits = prefill(model, prompt, cache)
    out: list[int] = []
    for _ in range(max_new_tokens):
        t = sample_token(logits, temperature, top_k, top_p, g)
        out.append(t)
        if len(out) < max_new_tokens:
            logits = decode_step(model, t, cache)
    return out


@torch.no_grad()
def speculative_generate(
    model: DeepSeekMini,
    draft: DeepSeekMini,
    prompt: list[int],
    max_new_tokens: int,
    num_draft: int = 3,
    seed: int | None = None,
) -> tuple[list[int], float]:
    """投机解码（draft-verify，贪心验证）。

    流程（每个验证轮）：
      1. 目标模型刚采出 token t（尚未喂进任何 cache）
      2. draft 自回归 propose k 个 token：吃 t 得 d1，吃 d1 得 d2 …
      3. 目标模型一次前向吃 [t, d1, …, d_k]，得到每个位置的 logits
      4. 逐位贪心验证：argmax(logits@位置i) == d_{i+1} 则接受；
         第一处不符 → 修正 token = argmax(logits)，crop 两边 cache 回滚
      5. 全部接受 → 用末位 logits 采出下一个 t

    贪心验证的漂亮性质：**输出与不用投机解码的贪心生成完全一致**
    （test_model.py 里有断言）。概率版验证（min(1, p/q) 接受 + residual
    重采样）留作练习，sampler.py:74-79 是 DeepInfer 的参考实现。

    教学简化：每轮验证失败后 draft 用 crop() 回滚（生产版在 BlockManager
    里做，还要处理抢占）。DSpark/置信度调度（按接受概率动态决定 k）
    留作进阶练习。
    """
    assert num_draft >= 1
    cache = KVCache()
    draft_cache = KVCache()
    logits = prefill(model, prompt, cache)
    _ = prefill(draft, prompt, draft_cache)

    out: list[int] = []
    accepted_total = 0
    proposed_total = 0
    t = sample_token(logits, temperature=0.0)

    while len(out) < max_new_tokens:
        out.append(t)
        if len(out) >= max_new_tokens:
            break

        # --- draft 自回归 propose k 个 ---
        k = min(num_draft, max_new_tokens - len(out))
        draft_tokens: list[int] = []
        d_logits = decode_step(draft, t, draft_cache)
        for _ in range(k):
            d = int(d_logits.argmax())
            draft_tokens.append(d)
            if len(draft_tokens) < k:
                d_logits = decode_step(draft, d, draft_cache)
        proposed_total += k

        # --- 目标模型一次吃 [t, d1, ..., d_k]，验证 + 为全接受情形备好末位 logits ---
        feed = [t] + draft_tokens
        x = torch.tensor([feed])
        logits_all, c_all, r_all = model.forward(x, cache.c_kv, cache.k_rope)
        cache.c_kv = c_all
        cache.k_rope = r_all

        accept_j = k  # 全接受时的默认
        for i, d in enumerate(draft_tokens):
            if int(logits_all[0, i].argmax()) != d:
                accept_j = i
                break

        if accept_j == k:
            # 全接受：末位 logits 采出下一 token；draft 也吃掉 d_k 对齐
            accepted_total += k
            accepted = draft_tokens
            t = int(logits_all[0, k].argmax())
            _ = decode_step(draft, draft_tokens[-1], draft_cache)
        else:
            # 前 accept_j 个接受，第一个不符处用目标 argmax 修正并回滚
            accepted_total += accept_j
            accepted = draft_tokens[:accept_j]
            t = int(logits_all[0, accept_j].argmax())
            new_len = cache.length - (k - accept_j)
            cache.crop(new_len)
            draft_cache.crop(new_len)

        out.extend(accepted)

    accept_rate = accepted_total / proposed_total if proposed_total else float("nan")
    return out[:max_new_tokens], accept_rate


def main() -> None:
    torch.manual_seed(0)
    target = DeepSeekMini(MiniConfig()).eval()
    draft = DeepSeekMini(MiniConfig(num_layers=1)).eval()

    prompt = [int(t) for t in torch.randint(0, 1024, (16,))]
    print(f"prompt tokens: {prompt}\n")

    # 1) 普通贪心生成
    t0 = time.perf_counter()
    plain = generate(target, prompt, max_new_tokens=32, temperature=0.0)
    t1 = time.perf_counter()
    print(f"普通贪心生成 32 tokens: {plain}")
    print(f"  耗时 {t1 - t0:.3f}s\n")

    # 2) 投机解码（贪心验证 → 输出必须与普通贪心一致）
    t0 = time.perf_counter()
    spec, accept_rate = speculative_generate(target, draft, prompt, max_new_tokens=32, num_draft=4)
    t1 = time.perf_counter()
    assert spec == plain, "贪心投机解码应当与普通贪心生成完全一致"
    print("投机解码 32 tokens（验证输出一致 OK）")
    print(f"  draft 接受率 {accept_rate:.0%}，耗时 {t1 - t0:.3f}s")

    # 3) MLA cache 省了多少
    from model import mha_cache_tokens_per_token, mla_cache_tokens_per_token
    cfg = MiniConfig()
    mla, mha = mla_cache_tokens_per_token(cfg), mha_cache_tokens_per_token(cfg)
    print(f"\nKV cache 每 token 每层：MLA {mla} 个元素 vs MHA {mha} 个元素"
          f"（省 {1 - mla / mha:.0%}）")


if __name__ == "__main__":
    main()
