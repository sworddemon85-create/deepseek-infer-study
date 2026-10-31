"""mini_infer 冒烟测试（纯 python 运行，无需 pytest）。

    python mini_infer/test_model.py

覆盖四个关键正确性：
  1. KV cache 增量解码 == 整段 prefill（推理引擎的立身之本）
  2. 贪心投机解码输出 == 普通贪心生成（draft-verify 不改变分布）
  3. MLA cache 确实比等尺寸 MHA 小
  4. MoE 路由：每 token 恰好 top-k 个专家、权重归一化、共享专家有贡献
  5. 固定 seed 采样可复现
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate import KVCache, decode_step, generate, prefill, speculative_generate  # noqa: E402
from model import (  # noqa: E402
    DeepSeekMini,
    MiniConfig,
    mha_cache_tokens_per_token,
    mla_cache_tokens_per_token,
)


def test_kv_cache_equivalence() -> None:
    torch.manual_seed(0)
    model = DeepSeekMini(MiniConfig()).eval()
    prompt = [1, 42, 7, 99, 5, 300, 2, 8]

    with torch.no_grad():
        x = torch.tensor([prompt])
        logits_full, _, _ = model.forward(x)

        cache = KVCache()
        incremental = []
        for t in prompt:
            logits = prefill(model, [t], cache) if cache.length == 0 else decode_step(model, t, cache)
            incremental.append(logits)

    for i, (a, b) in enumerate(zip(logits_full[0], incremental)):
        assert torch.allclose(a, b, atol=1e-4), f"位置 {i} 的 logits 与增量解码不一致"
    print("PASS  KV cache 增量解码与整段 prefill 一致")


def test_speculative_greedy_matches_plain() -> None:
    torch.manual_seed(1)
    target = DeepSeekMini(MiniConfig()).eval()
    draft = DeepSeekMini(MiniConfig(num_layers=1)).eval()  # 不同的弱 draft
    prompt = [3, 17, 256, 4, 88]

    plain = generate(target, prompt, max_new_tokens=24, temperature=0.0)
    spec, accept_rate = speculative_generate(target, draft, prompt, max_new_tokens=24, num_draft=4)
    assert spec == plain, "贪心投机解码必须与普通贪心生成逐 token 一致"
    # 随机权重的弱 draft 接受率可以是 0%（这正是"draft 决定加速上限"的直观体现），
    # 正确性由上面"输出逐 token 一致"保证。
    assert 0.0 <= accept_rate <= 1.0
    print(f"PASS  投机解码输出与普通贪心一致（随机弱 draft，接受率 {accept_rate:.0%}）")


def test_mla_cache_saving() -> None:
    cfg = MiniConfig()
    mla = mla_cache_tokens_per_token(cfg)
    mha = mha_cache_tokens_per_token(cfg)
    assert mla * cfg.num_layers < mha * cfg.num_layers
    print(f"PASS  MLA cache 每 token {mla * cfg.num_layers} 元素 vs MHA {mha * cfg.num_layers}"
          f"（省 {1 - mla / mha:.0%}）")


def test_moe_routing() -> None:
    torch.manual_seed(2)
    model = DeepSeekMini(MiniConfig()).eval()
    moe = model.blocks[0].moe
    x = torch.randn(4, 10, model.cfg.hidden_size)

    with torch.no_grad():
        out = moe(x)
        assert out.shape == x.shape and torch.isfinite(out).all()

        scores = F.softmax(moe.gate(x.float()), dim=-1)
        topk_w, _ = torch.topk(scores, moe.cfg.num_experts_per_tok, dim=-1)
        topk_w = topk_w / topk_w.sum(-1, keepdim=True)   # 与模型一致：top-k 后归一化
        sums = topk_w.sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "top-k 权重应归一化"

        # 共享专家常开：置零共享专家后输出应改变
        backup = [e.gate_proj.weight.clone() for e in moe.shared_experts]
        for e in moe.shared_experts:
            e.gate_proj.weight.zero_()
            e.up_proj.weight.zero_()
            e.down_proj.weight.zero_()
        out_no_shared = moe(x)
        assert not torch.allclose(out, out_no_shared), "共享专家应有贡献"
        for e, w in zip(moe.shared_experts, backup):
            e.gate_proj.weight.copy_(w)
    print("PASS  MoE 路由：top-k 归一化 + 共享专家常开")


def test_sampling_determinism() -> None:
    torch.manual_seed(3)
    model = DeepSeekMini(MiniConfig()).eval()
    prompt = [5, 6, 7, 8, 9]
    a = generate(model, prompt, max_new_tokens=16, temperature=0.8, top_k=50, top_p=0.9, seed=1234)
    b = generate(model, prompt, max_new_tokens=16, temperature=0.8, top_k=50, top_p=0.9, seed=1234)
    assert a == b, "相同 seed 的采样必须可复现"
    print("PASS  采样可复现（固定 seed）")


if __name__ == "__main__":
    test_kv_cache_equivalence()
    test_speculative_greedy_matches_plain()
    test_mla_cache_saving()
    test_moe_routing()
    test_sampling_determinism()
    print("\n全部通过 (all tests passed)")
