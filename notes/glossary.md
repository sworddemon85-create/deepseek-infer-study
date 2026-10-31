# 术语表：V4.1-Flash 新概念速查

读 DeepInfer 代码前先混个脸熟。均为精读代码归纳，非官方定义。

## 注意力相关

| 术语 | 含义 | 关键代码 |
|---|---|---|
| MLA | Multi-head Latent Attention：KV 压缩到低秩隐向量再升维，KV cache 只存压缩量 | V3 起的基础，FlashMLA 内核 |
| YoCo | 分层 KV 共享：`n_prefill_layer` 把层切两半，consumer 层的 KV 从 shared hidden 投影 | `yoco.py`、`attn.py:245-260` |
| SWA | 滑窗注意力（sliding window attention），有限窗口内计算注意力 | `attn.py` `use_swa` |
| attention sink | 给每个注意力头一个可学习的 sink 偏置（`sink_mode == "swa"`），稳定滑窗注意力 | `attn.py:266-271` |
| MQAIndexer | MQA 形式的稀疏索引器：先选 top-k token 再主注意力（DSA 思路） | `mla_attn.py:100` |
| KV 压缩比 | `kv_compress_ratio`：KV 投影带 Gate + 低秩压缩，`is_c1` = 不压缩 | `attn.py:308-373` |
| SharedLayerTopology | KV 源层 / 索引源层 / 索引 k 源层的共享拓扑，三者可解耦 | `attn.py:251-260` |
| 有界回放（bounded replay） | prefill 时只回放有限后缀而非全序列，扣掉 SWA 窗口即可 | README feature；`DEEPINFER_YOCO_*` |

## 模型结构

| 术语 | 含义 | 关键代码 |
|---|---|---|
| Engram | n-gram 哈希压缩记忆：每层一张哈希表，token 序列的 n-gram 哈希后注入嵌入 | `engram*.py`、`tile_kernels.engram.engram_hash` |
| HC / MHC | hidden 压缩头（`HCHead`、`hc_mult`、`hc_shifted`）与跨层混合 | `hc*.py`、`mhc_*.py` |
| DFlash | 投机解码 draft 模块（MTP 层），配 ConfidenceHead | `mtp/dflash.py` |
| DSpark | DFlash 相关的目标层特征方案（`dflash_cfg.dspark_target_layer_ids`） | `deepseek.py:72-74` |
| CP / EP / DP | 上下文并行 / 专家并行 / 数据并行；`world = dp × cp` 须整除 ep | `offline_inference.py:27-29` |
| EPLB | Expert Parallelism Load Balancer：专家负载均衡（冗余专家迁移） | `eplb/`、[官方仓库](https://github.com/deepseek-ai/EPLB) |

## 引擎 / 调度

| 术语 | 含义 | 关键代码 |
|---|---|---|
| ZOS | zero-overhead scheduling：CPU 调度与 GPU 执行重叠的异步调度 | `csrc/engine/zos.*` |
| BlockManager | KV 块 / state 序列的块管理（pin/unpin、clone/zero） | `csrc/engine/block_manager.hpp` |
| 批不变推理 | 结果与 batch 组成无关（`DEEPINFER_BATCH_INVARIANT=1`，仅 NVIDIA） | README feature |
| 确定性采样 | 相同输入 + 固定 `seq_seed` → 相同输出（需关 MTP） | README feature |
| SO | structured output：正则/JSON/EBNF 约束生成 | `structure_output/`、`llm_engine.py:910-928` |
| DeferredRelease | 延迟一轮回收资源，与 GPU 计算 overlap | `llm_engine.py:67-119` |

## 采样 / 投机解码

| 术语 | 含义 | 关键代码 |
|---|---|---|
| nextn / MTP | 多 token 预测层：一次 propose 多个 draft token | `mtp/` |
| 置信度调度 | 按 ConfidenceHead 预测的接受概率决定验证长度 | `dflash.py:46-79` |
| residual probability | 拒绝位重采样分布 `max(p_target - q_draft, 0)` 归一化 | `sampler.py:74-79` |
| force_topk | logprobs 只存 top-k（默认 4096）省内存 | `dflash.py:102-106` |
| accept_rate | draft 接受率，`accept_tok / draft_tok` 聚合 | `offline_inference.py:82-86` |
