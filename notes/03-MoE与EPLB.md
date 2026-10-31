# 03 · MoE 与 EPLB

> 核心文件：`deepinfer/model_executor/models/v4_modules/moe.py`、
> `mlp.py`、`deepinfer/model_executor/eplb/`。

## 专家计数语义（moe.py 文件头注释，值得背下来）

```
logical  = cfg.moe.n_routed_experts          # 门控可选范围
extra    = cfg.moe.n_redundant_experts       # 冗余专家（默认 32，EPLB 迁移用）
physical = logical + extra                   # 实际参数行数，须整除 ep_size
local    = physical / ep_size                # 本 rank 持有的专家数
```

为什么需要冗余专家？热门专家（比如代码/常见词）会被过量 token 命中，
单卡成为瓶颈。EPLB 把热门专家复制成多份物理副本分散到不同卡，
门控逻辑层看到的还是 `logical` 个"逻辑专家"，路由表把逻辑专家映射到
物理副本——这就是"逻辑/物理解耦"。

## 门控：TopKGate（moe.py:40-67）

```python
scoring_func  == "sqrtsoftplus"     # 打分函数（assert 固定）
infer_topk_method == "noaux_tc"     # top-k 方法（assert 固定）
norm_topk_prob                    # top-k 后归一化
e_score_correction_bias           # FP32 负载修正 bias
e_score_correction_bias_vl        # 图像 token 的另一套 bias（V4.1 多模态路由）
Wgate                             # bf16，不参与 FP8 量化
```

三个细节：

1. **打分与选择解耦**：`noaux_tc` 的含义是打分值本身不参与排序，
   用 `score + bias` 排序选 top-k，但混合权重仍用原始 score——
   这样 bias 只影响"谁被选中"，不扭曲选中专家的贡献比例，
   同时起到负载均衡的作用（DeepSeek-V3 论文中的 auxiliary-loss-free 策略）。
2. **文本/图像双 bias**：多模态 token 的专家偏好不同，路由修正是分开的。
3. **门控永远 bf16**：路由是每 token 的标量运算，不值得上 FP8。

## 专家容器：FusedExperts（moe.py:70-113）

- 路由专家：`ParallelRoutedLinear`（按 EP 切分，本 rank 只存 local 个）。
- 共享专家：`ReplicatedSharedExpert`——每个 rank 都有，`n_shared_experts`
  个专家只是沿 hidden 维拼接成一个宽 Linear（不产生多个子模块），
  FP8 可选。
- 激活：SwiGLU，带 `swiglu_clamp_value`（`cfg.swiglu_limit`）上限截断。

## EPLB 怎么接进来

- 入口脚本默认 `enable_eplb=True`，通过 `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS`
  （默认 32）传冗余数——注意它只是**下限**：引擎会把
  `logical + extra` 向上取整到能被 `ep_size` 整除（backbone 与 DFlash 分别取整）。
- 统计刷新有周期（`DEEPINFER_MOE_LB_STAT_INTERVAL`），伴随 EP all_gather
  通信抖动（benchmark 笔记里提到十几 ms），所以 benchmark 的测量窗口
  会刻意避开 LB 刷新阶段——测量与系统开销共存时的实用技巧。

## 对照 mini_infer

`mini_infer/model.py` 的 `DeepseekMoE` 实现了教学版门控：softmax 打分 →
top-k → 归一化 → routed_scaling，外加一个常开共享专家。
bias 负载修正与冗余专家迁移在生产系统才有意义，教学版略去，
但在注释里标注了 DeepInfer 对应位置。
