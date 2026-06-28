# CRT 实验指导手册

> 面向双 A800 GPU 服务器的训练和推理实验指南

## 1. 环境搭建

### 1.1 安装

```bash
# 克隆并安装
git clone <your-repo-url> crt && cd crt
pip install -e ".[full]"

# 验证 GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
# 预期输出: CUDA: True, GPUs: 2
```

### 1.2 准备数据集

```bash
python3 scripts/download_data.py --all
```

生成 4 个数据集到 `data/` 目录：

| 数据集 | 节点 | 边 | 用途 |
|--------|------|-----|------|
| sachs | 11 | 17 | 蛋白质信号通路，因果发现金标准 |
| asia | 8 | 8 | 贝叶斯网络经典 benchmark |
| alarm | 37 | 47 | 医疗监控，中等规模 |
| medical_demo | 8 | 7 | 语义推理 demo（含文本场景） |

---

## 2. 实验一：因果图发现（合成数据）

验证 CRT 的核心能力——从数据中发现因果结构。

### 2.1 单卡训练

```bash
# 基础实验：10 节点，线性方程
python3 scripts/train.py \
    --num-nodes 10 \
    --num-graphs 5000 \
    --equation-type linear \
    --epochs 200 \
    --batch-size 64 \
    --lr 1e-3 \
    --device cuda:0 \
    --log-dir runs/exp1_linear \
    --checkpoint-dir checkpoints/exp1_linear
```

### 2.2 双卡 DDP 训练（推荐）

```bash
# 双 A800 分布式训练
torchrun --nproc_per_node=2 scripts/train.py \
    --num-nodes 10 \
    --num-graphs 10000 \
    --equation-type linear \
    --epochs 200 \
    --batch-size 64 \
    --lr 1e-3 \
    --log-dir runs/exp1_ddp \
    --checkpoint-dir checkpoints/exp1_ddp
```

### 2.3 消融实验矩阵

建议按以下维度做对比实验：

```bash
# 1. 方程类型对比
for eq in linear nonlinear mixed; do
    torchrun --nproc_per_node=2 scripts/train.py \
        --num-nodes 10 --num-graphs 5000 --equation-type $eq \
        --epochs 200 --batch-size 64 \
        --log-dir runs/ablation_eq_$eq \
        --checkpoint-dir checkpoints/ablation_eq_$eq
done

# 2. 节点规模对比
for n in 5 10 20 30 50; do
    torchrun --nproc_per_node=2 scripts/train.py \
        --num-nodes $n --num-graphs 5000 --epochs 200 --batch-size 64 \
        --log-dir runs/ablation_nodes_$n \
        --checkpoint-dir checkpoints/ablation_nodes_$n
done

# 3. 迭代次数（改图轮数）对比
for it in 1 2 3 5 8; do
    torchrun --nproc_per_node=2 scripts/train.py \
        --num-nodes 10 --num-graphs 5000 --num-iterations $it \
        --epochs 200 --batch-size 64 \
        --log-dir runs/ablation_iter_$it \
        --checkpoint-dir checkpoints/ablation_iter_$it
done
```

### 2.4 评估

```bash
python3 scripts/evaluate.py \
    --checkpoint checkpoints/exp1_ddp/best.pt \
    --num-eval-graphs 500 \
    --output-dir eval_output/exp1_ddp \
    --device cuda:0

# 查看指标
cat eval_output/exp1_ddp/metrics.json

# 查看可视化
ls eval_output/exp1_ddp/figures/
```

### 2.5 指标说明

| 指标 | 含义 | 好的结果 |
|------|------|----------|
| SHD | 结构汉明距离（越低越好） | 10 节点图 < 5 |
| F1 | 边发现的 F1 值 | > 0.7 |
| TPR | 真阳性率（找到了多少真实边） | > 0.8 |
| FDR | 假发现率（多少发现是错的） | < 0.2 |

---

## 3. 实验二：层次化因果图

测试多层抽象推理能力。

```bash
# 三层：8(微观) + 4(中观) + 2(宏观) = 14 节点
torchrun --nproc_per_node=2 scripts/train.py \
    --hierarchical \
    --level-sizes 8 4 2 \
    --num-graphs 5000 \
    --epochs 200 \
    --batch-size 64 \
    --log-dir runs/exp2_hierarchical \
    --checkpoint-dir checkpoints/exp2_hierarchical

# 对比：同样 14 个节点的平面图
torchrun --nproc_per_node=2 scripts/train.py \
    --num-nodes 14 \
    --num-graphs 5000 \
    --epochs 200 \
    --batch-size 64 \
    --log-dir runs/exp2_flat \
    --checkpoint-dir checkpoints/exp2_flat
```

---

## 4. 实验三：双模式语义推理

### 4.1 Tool 模式训练

CRT 作为外部工具——接收文本，输出因果链。

```bash
# 单卡
python3 scripts/train_unified.py \
    --mode tool \
    --dataset medical_demo \
    --num-repeats 500 \
    --epochs 100 \
    --batch-size 32 \
    --embed-dim 128 \
    --query-dim 64 \
    --initial-k 4 \
    --max-steps 4 \
    --hidden-dim 64 \
    --lr 5e-4 \
    --device cuda:0 \
    --log-dir runs/exp3_tool \
    --checkpoint-dir checkpoints/exp3_tool

# 双卡
torchrun --nproc_per_node=2 scripts/train_unified.py \
    --mode tool \
    --dataset medical_demo \
    --num-repeats 1000 \
    --epochs 100 \
    --batch-size 32 \
    --embed-dim 128 \
    --query-dim 64 \
    --initial-k 4 \
    --max-steps 4 \
    --hidden-dim 64 \
    --lr 5e-4 \
    --log-dir runs/exp3_tool_ddp \
    --checkpoint-dir checkpoints/exp3_tool_ddp
```

### 4.2 Embedded 模式训练

因果推理嵌入 Transformer 层——用因果图约束 attention。

```bash
# 单卡
python3 scripts/train_unified.py \
    --mode embedded \
    --dataset medical_demo \
    --num-repeats 500 \
    --epochs 100 \
    --batch-size 32 \
    --d-model 256 \
    --num-heads 8 \
    --num-layers 6 \
    --d-ff 512 \
    --hidden-dim 64 \
    --causal-weight 0.8 \
    --lr 3e-4 \
    --device cuda:0 \
    --log-dir runs/exp3_embedded \
    --checkpoint-dir checkpoints/exp3_embedded

# 双卡
torchrun --nproc_per_node=2 scripts/train_unified.py \
    --mode embedded \
    --dataset medical_demo \
    --num-repeats 1000 \
    --epochs 100 \
    --batch-size 32 \
    --d-model 256 \
    --num-heads 8 \
    --num-layers 6 \
    --d-ff 512 \
    --hidden-dim 64 \
    --lr 3e-4 \
    --log-dir runs/exp3_embedded_ddp \
    --checkpoint-dir checkpoints/exp3_embedded_ddp
```

### 4.3 推理测试

```bash
# Tool 模式：交互式因果问答
python3 scripts/infer.py --mode tool \
    --checkpoint checkpoints/exp3_tool/best.pt \
    --device cuda:0

# Tool 模式：单条推理
python3 scripts/infer.py --mode tool \
    --checkpoint checkpoints/exp3_tool/best.pt \
    --context "Patient has fever and cough. Got wet in rain 3 days ago." \
    --query "What caused the fever?" \
    --device cuda:0

# Embedded 模式：文本生成
python3 scripts/infer.py --mode embedded \
    --checkpoint checkpoints/exp3_embedded/best.pt \
    --prompt "The patient presents with" \
    --max-tokens 100 \
    --device cuda:0
```

---

## 5. 监控训练

### TensorBoard

```bash
# 在服务器上启动
tensorboard --logdir runs/ --port 6006 --bind_all

# 在本地浏览器打开（通过 SSH 隧道）
# ssh -L 6006:localhost:6006 <server>
# 打开 http://localhost:6006
```

### 实时查看训练日志

所有训练脚本都会打印每个 epoch 的指标：
```
Epoch 042 | train_loss=0.3214 | val_loss=0.3891 | val_shd=4.2 | val_f1=0.723
```

---

## 6. A800 双卡性能参考

| 实验 | 节点数 | 数据量 | 单卡时间/epoch | 双卡时间/epoch |
|------|--------|--------|---------------|---------------|
| 合成数据 | 10 | 5000 图 | ~15s | ~8s |
| 合成数据 | 20 | 5000 图 | ~40s | ~22s |
| 合成数据 | 50 | 5000 图 | ~3min | ~1.5min |
| Tool 模式 | 8 | 500 场景 | ~5s | ~3s |
| Embedded 模式 | 8 | 500 场景 | ~8s | ~5s |

> 以上为估算值，实际时间取决于 batch_size 和模型配置。

---

## 7. 推荐实验顺序

```
第一周：实验一（合成数据因果发现）
  → 验证 CRT 核心能力，调通训练流程
  → 关键指标：SHD < 5, F1 > 0.7

第二周：实验二（层次化 vs 平面图对比）
  → 验证多层抽象是否带来增益
  → 关键指标：层次化在复杂图上 SHD 更低

第三周：实验三 Tool 模式
  → 验证语义理解能力
  → 关键指标：正确识别激活节点，发现因果链

第四周：实验三 Embedded 模式
  → 验证因果约束对 attention 的影响
  → 关键指标：DAG penalty 下降，LM loss 下降
```

---

## 8. 文件结构总览

```
crt/
├── crt/
│   ├── graph/
│   │   ├── causal_graph.py          # 可微分因果图引擎
│   │   └── hierarchical_graph.py    # 层次化因果图
│   ├── transformer/
│   │   └── causal_attention.py      # 因果约束注意力
│   ├── integration/
│   │   ├── adaptive_model.py        # 闭环改图模型
│   │   ├── sparse_reasoning.py      # 稀疏推理 + 自适应停止
│   │   ├── semantic_reasoning.py    # 语义编码器 + 文本数据集
│   │   └── dual_mode.py            # Tool/Embedded 双模式
│   ├── utils/
│   │   ├── metrics.py               # SHD, F1, TPR/FDR
│   │   └── visualization.py         # 图对比、热力图、曲线
│   └── data/
│       └── synthetic.py             # 合成 SCM 数据生成
├── scripts/
│   ├── train.py                     # 因果发现训练（合成数据）
│   ├── train_unified.py             # 双模式统一训练
│   ├── evaluate.py                  # 评估 + 可视化
│   ├── infer.py                     # 推理（交互/批量）
│   └── download_data.py             # 数据集下载
├── data/
│   ├── sachs/                       # Sachs 蛋白质数据
│   ├── asia/                        # Asia 贝叶斯网络
│   ├── alarm/                       # ALARM 医疗监控
│   └── medical_demo/                # 语义推理 demo
└── docs/
    └── experiment_guide.md          # 本文档
```
