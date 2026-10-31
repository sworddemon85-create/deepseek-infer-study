# deepseek-infer-study

DeepSeek 开源推理框架 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 的源码精读笔记，以及一个配套的、从零手写的迷你 LLM 推理引擎，把"读大厂框架"和"自己动手写"放在一个仓库里。

> 本仓库是个人学习项目，与 DeepSeek 官方无关。所有笔记基于 DeepInfer main 分支 2026-10-27 快照，行号均对应该快照。

## 仓库结构

```
deepseek-infer-study/
├── notes/                     # DeepInfer 源码精读笔记（按模块分章）
│   ├── 00-总览与阅读路线.md        # 分层架构 + 精读顺序
│   ├── 01-引擎与调度.md            # llm_engine.step() 与 C++ 调度器
│   ├── 02-模型架构-V4.1-Flash.md   # DeepseekModel forward 与新概念
│   ├── 03-MoE与EPLB.md            # 专家并行、负载均衡、门控细节
│   ├── 04-投机解码-DFlash.md       # MTP + 置信度调度验证
│   ├── 05-算子生态.md              # FlashMLA / DeepGEMM / DeepEP / TileKernels
│   └── glossary.md               # 术语表（V4.1 新概念速查）
├── mini_infer/                # 迷你推理引擎（纯 PyTorch，可读可跑）
│   ├── model.py               # MLA 注意力 + MoE + RoPE，带中文注释
│   ├── generate.py            # KV cache、采样、投机解码（draft-verify）
│   ├── test_model.py          # 冒烟测试：cache 一致性、投机验证正确性
│   └── README.md              # 原理讲解：MLA 为什么省 KV cache
└── README.md
```

## 快速开始

```bash
# 只需 torch（CPU 版即可）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 跑冒烟测试（验证 KV cache 一致性与投机解码逻辑）
python mini_infer/test_model.py

# 随机权重生成一段文本，观察 MLA/MoE/投机解码的机制
python mini_infer/generate.py
```

迷你引擎的权重是随机初始化的——它演示的是**推理机制**（KV cache 压缩、专家路由、draft-verify 循环），不是语言能力。原理讲解见 [mini_infer/README.md](mini_infer/README.md)。

## 笔记阅读路线

1. [notes/00-总览与阅读路线.md](notes/00-总览与阅读路线.md) —— 先建立全局图景
2. [notes/glossary.md](notes/glossary.md) —— V4.1 的新名词先混个脸熟（YoCo / DFlash / Engram / HC …）
3. 01 引擎 → 02 模型 → 03 MoE → 04 投机解码 → 05 算子，逐章深入
4. 每读完一章，去 `mini_infer/` 找对应的最小实现对照着看

## 相关官方仓库

| 仓库 | 内容 |
|---|---|
| [DeepInfer](https://github.com/deepseek-ai/DeepInfer) | 生产级推理框架（本项目精读对象） |
| [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) | 含几百行极简 `inference/` 推理 demo，入门首选 |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | MLA 高效内核 |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | FP8 通用矩阵乘 |
| [DeepEP](https://github.com/deepseek-ai/DeepEP) | 专家并行通信库 |
| [TileKernels](https://github.com/deepseek-ai/TileKernels) | TileLang 算子库 |

## License

MIT，见 [LICENSE](LICENSE)。笔记与代码仅供学习交流。
