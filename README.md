# CRT - Causal Reasoning Transformer

将因果推理能力注入 Transformer 架构的研究项目。

## 研究方向

### Path 1: Perception + Reasoning 分离架构
Transformer 做感知（编码非结构化输入），因果图做结构化推理。

### Path 2: Causal Attention
在 Transformer 的 attention 机制中注入因果约束，打破"相关性≠因果性"。

### Path 3: Hypothesis-Verification Loop
Transformer 生成因果假设，因果图引擎验证，验证结果作为 reward 信号。

## 项目结构

```
crt/
├── crt/                    # 核心库
│   ├── graph/              # 因果图引擎（可微分）
│   ├── transformer/        # Transformer 变体
│   ├── integration/        # 三条路径的融合模块
│   └── utils/              # 工具函数
├── experiments/            # 实验脚本和配置
├── data/                   # 数据集
├── docs/                   # 研究文档
├── scripts/                # 训练/评估脚本
└── tests/                  # 测试
```

## 开始

```bash
pip install -e .
```
