# 04 · 投机解码：DFlash 与置信度调度

> 核心文件：`deepinfer/model_executor/models/mtp/dflash.py`、
> `deepinfer/model_executor/layers/sampler.py`、`layers/mtp_info_pool.py`。
> 环境变量：`DEEPINFER_ENABLE_MTP_VARLEN=1` 开启；
> `DEEPINFER_USE_NEXTN_DECODE` 控制草稿长度（默认 5，置 0 关闭）。

## 为什么是 DFlash 而不是普通 MTP

普通 MTP：draft 模型一次 propose 固定 k 个 token，目标模型整段验证。
问题是接受率随位置衰减——后面的 draft 往往是垃圾，验证却白算。

DFlash 的不同：**ConfidenceHead 给每个 draft token 预测"接受概率"**，
调度器根据置信度动态决定这一步验证多长（confidence-scheduled
verification，README 的 feature 列表原文）。省下的是对低置信 draft 的
无效验证。

## ConfidenceHead（dflash.py:46-79）

```python
class ConfidenceHead(nn.Module):
    """Predict the acceptance probability of each draft token."""
    # 输入: hidden + markov_emb → 线性到 1 维 logit → /temperature → sigmoid
    # 温度按解码步数取值（DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE）
    # 前向用的是 fused kernel: confidence_head_fused (tilelang_ops/dflash.py)
```

`markov_emb` 来自 markov_rank 维嵌入——DFlash 带马尔可夫式的
历史压缩特征（对应 `dflash_cfg.dspark_target_layer_ids` 的目标层特征）。

## 特征攒取与 CUDA graph（deepseek.py:220-223）

主模型 forward 顺手把各目标层 hidden 拼进 `dflash_input`，
然后 `dflash.update_main_cache(...)` 写入 DFlash 的 main KV cache。
注释点明动机：**decode 时随主模型 CUDA graph 一并捕获，
省去逐 kernel 的 CPU 发射开销**。CUDA graph 的 padding 行落在
BlockManager 保留的 dummy slot（state 0）。

## Sampler 的完整流程（sampler.py:59-80 的文档串，原文极清晰）

Prefill 阶段：
1. 只采样每个 prompt 最后一个 token 的输出
2. DFlash 的 main KV cache 已由主模型 forward 更新，无额外 draft prefill

Decode 阶段（开 MTP 时）：
1. **验证**上一轮 draft：逐步比较 draft token 在新旧分布下的 logprob，
   按 speculative decoding 的接受/拒绝规则决定
2. 被接受的 draft token 作为输出
3. 第一个被拒绝的位置用 **residual probability** 重采样
   （`residual = max(p_target - q_draft, 0)` 归一化后采样）
4. 全部接受则在最后一个 draft token 之后正常采样
5. 用 nextn 模块对输出 token 做一步 forward，预测下一轮 draft 及其 logprobs

## 省内存的 logprobs

`logprobs_dim = vocab_size`，但开 `DEEPINFER_SAMPLER_FORCE_TOPK`（默认 4096）
后只存 top-4096 的 logprobs 并额外记录 index——验证只需要
draft token 在目标分布下的概率，不需要全词表。

## 对照 mini_infer

`mini_infer/generate.py` 实现了教学版投机解码：1 层 draft 模型 + 贪心
验证（argmax 相等则接受，否则回退重采样）。分布版 residual 重采样
的数学在 `test_model.py` 里有正确性断言。置信度调度属于生产级特性，
教学版在 README 里留了"如何扩展"的思考题。
