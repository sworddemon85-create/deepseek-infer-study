# mini_infer：从零写一个迷你 DeepSeek 推理引擎

这是一个**教学用**迷你推理引擎：不追求性能，追求每一行都能看懂，
把 DeepSeek 系模型的推理机制拆到最小可运行粒度。配套 `notes/` 里的
DeepInfer 精读笔记——那边是生产级实现，这边是机制的最小化骨架。

## 文件

| 文件 | 内容 |
|---|---|
| `model.py` | RMSNorm / RoPE / **MLA** / **MoE** / TransformerBlock / DeepSeekMini |
| `generate.py` | KVCache、增量解码、采样（temperature/top-k/top-p）、**投机解码** |
| `test_model.py` | 4 个正确性断言（cache 一致性、投机解码不变性、cache 省量、MoE 路由） |

## 1. MLA：KV cache 为什么能小这么多

标准 MHA 里，每个注意力头都要缓存自己的 K 和 V：

```
cache_MHA = heads × (k_dim + v_dim)          # 每 token 每层
```

MLA（Multi-head Latent Attention）的思路：把 KV **压缩成一个低秩向量
c_KV 再按需升维**，cache 只存压缩量：

```
c_KV  = W_DKV · h              # [kv_lora_rank]     ← 存这个
k_rope= RoPE(W_KR · h)         # [rope_dim]         ← 和这个（共享位置 key）
k_i   = W_UK_i · c_KV          # 用到时再升维（每 head 一份投影）
v_i   = W_UV_i · c_KV
```

本仓库默认配置下（`model.py: MiniConfig`）：

```
cache_MLA = kv_lora_rank + rope = 32 + 16 = 48
cache_MHA = heads × (nope + rope + v) = 4 × 48 = 192
→ 省 75%
```

跑 `test_model.py` 的 `test_mla_cache_saving` 看数字；跑
`test_kv_cache_equivalence` 确认增量解码与整段前向逐位一致。

**生产版在做什么**（对应 `notes/02` 与 DeepInfer `attn.py`）：
权重吸收把 W_UK/W_UV 折进 Q/O 侧，升投影就不再出现在每步前向里；
KV 本身再压缩（`kv_compress_ratio` + Gate）、FP8 量化、
稀疏索引（MQAIndexer）选 top-k token、YoCo 跨层共享——每一条都是
在这 48 个数的基础上继续抠。

## 2. MoE：路由、共享专家与负载

每个 token 由门控打分选 top-k 个专家加权求和，另加**常开**的共享专家：

```python
scores = softmax(gate(x))
w, idx = topk(scores, k)          # 每 token 只激活 k 个专家
w = w / w.sum()                   # 归一化
out = Σ w_i · expert_i(x) + shared(x)
```

生产版（DeepInfer `v4_modules/moe.py`）的三个关键差异：

1. 打分用 `sqrtsoftplus`、选择用 `noaux_tc` + FP32 负载修正 bias——
   bias 只影响"谁被选中"，不扭曲选中后的贡献权重（auxiliary-loss-free 均衡）；
2. 专家分**逻辑/物理**两层：冗余副本（默认 32 个）由 EPLB 迁移热门专家；
3. 专家按 EP 切到多卡，token 走 DeepEP 的 all-to-all。

## 3. 投机解码：draft-verify 循环

```
1. 目标模型采出 token t（尚未进 cache）
2. draft 自回归 propose k 个：吃 t 得 d1，吃 d1 得 d2 …
3. 目标模型一次前向吃 [t, d1 … dk] —— 并行算出 k+1 个位置的 logits
4. 逐位验证：argmax(logits@i) == d_i 则接受；第一处不符 → 用目标 argmax
   修正，两边 cache crop 回滚
5. 全接受 → 用末位 logits 采出下一个 t，继续下一轮
```

贪心验证有个漂亮性质：**输出与不用投机的贪心生成逐 token 一致**
（`test_speculative_greedy_matches_plain` 断言的就是这个）。
GPU 一次吃 k+1 个 token 的开销 ≈ 一次吃 1 个，却可能白拿 k 个正确 token——
这就是加速的来源，也是 DeepInfer benchmark 里
`decode throughput = avg_SPS × batch × (1 + nextn)` 的那个 `1 + nextn`。

生产版 DFlash（`notes/04`）在这之上加了两件事：ConfidenceHead 预测每个
draft 的接受概率，**按置信度决定这一轮验证多长**（省掉对低置信 draft 的
无效验证）；DFlash 的 KV 由主模型 forward 顺手写、随 CUDA graph 捕获。

## 4. 刻意没做的事

FP8/量化、张量并行、专家并行、CUDA graph、连续批处理、分页 KV、
结构化输出、YOoC 共享、稀疏索引、Engram、HC 头……每一条都值得做完
再读 DeepInfer 的对应章节——它们就是 `notes/` 的目录。

## 5. 扩展练习（由易到难）

1. 概率版验证：把贪心验证换成 `min(1, p_target(d)/q_draft(d))` 接受 +
   residual 分布 `max(p-q, 0)` 重采样（参考 `notes/04` 的公式）。
2. 权重吸收版 MLA：把 W_UK 折进 Q、W_UV 折进 O，验证数值不变。
3. 置信度调度：给 draft 加一个小 ConfidenceHead，按接受概率动态截断 draft 长度。
4. 负载修正门控：加 FP32 bias 参与排序但不参与加权，观察专家命中分布变化。
