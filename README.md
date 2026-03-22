# ConColl: Sequential Multi-Stage Vulnerability Detection

复现 EMNLP 2025 论文 "A Sequential Multi-Stage Approach for Code Vulnerability Detection via Confidence- and Collaboration-based Decision Making"

## Overview

ConColl 通过基于置信度的顺序决策，动态选择三种检测策略：

```
代码 → Stage 1 (Direct)
        ↓ C.S. < threshold
       Stage 2 (RAG)
        ↓ C.S. < threshold
       Stage 3 (Multi-Agent)
```

**置信度分数 (C.S.)** = P(top-1) - P(top-2)

**接受条件**：C.S. ≥ threshold **且** top-1 token 为 "Yes" 或 "No"

### 模型支持与测试结果

| 模型 | logprobs | 置信度 | 样本数 | Accuracy | Precision | Recall | F1 | Stage 分布 |
|------|----------|--------|--------|----------|-----------|--------|-----|------------|
| **MiniMax** | ✗ | 固定0.15 | 4 | 0.75 | 1.00 | 0.50 | 0.67 | S1:0 / S2:4 / S3:4 |
| **DeepSeek** | ✓ | 真实计算 | 20 | 0.55 | 0.53 | 1.00 | 0.69 | S1:20 / S2:0 / S3:0 |
| **GPT-4** | ✓ | 真实计算 | - | - | - | - | - | - |

**分析**:
- **MiniMax**: 无 logprobs，使用固定置信度 0.15，所有样本自动进入 Stage 2/3 进行详细分析，精确率 100%（预测为漏洞的都对了）
- **DeepSeek**: 通过 OpenAI 兼容 API 返回 logprobs，置信度计算正常；所有样本在 Stage 1 被接受（置信度 > 0.3），因为模型倾向于输出 "VULNERABLE"，首词概率极高；召回率 100% 但精确率仅 52.63%，模型倾向于将所有样本预测为漏洞

## 快速开始

```bash
# 基本运行
uv run run_concoll.py --max-samples 20

# 常用选项组合
uv run run_concoll.py --max-samples 100 --confidence-threshold 0.5    # 高阈值，更多样本进入 Stage 2/3
uv run run_concoll.py --test-data                                       # 使用合成测试数据
uv run run_concoll.py --simulate                                       # 模拟模式（固定分流比例）
```

## 配置

### 置信度阈值

控制 Stage 1 的置信度阈值，影响样本分流：

| 阈值 | 效果 |
|------|------|
| 高 (0.7) | 更多样本进入 Stage 2/3，精度更高，成本更高 |
| 中 (0.3) | 平衡模式（默认） |
| 低 (0.1) | 更多样本在 Stage 1 被接受，成本低 |

**配置方式**（优先级从高到低）：
1. 命令行: `--confidence-threshold 0.5`
2. 环境变量: `CONFIDENCE_THRESHOLD=0.5`
3. .env 文件: `CONFIDENCE_THRESHOLD=0.5`

### 置信度计算原理

```python
# GPT-4/DeepSeek logprobs 格式
logprobs = [
    {"token": "Yes", "logprob": -0.1},   # P ≈ 0.90
    {"token": "No",  "logprob": -2.3}    # P ≈ 0.10
]
# 置信度 = 0.90 - 0.10 = 0.80

# MiniMax/GLM (无 logprobs)
# 使用固定低置信度 0.15，确保样本进入 Stage 2/3 进行更详细分析
```

### 切换模型

在 `.env` 中配置：

```bash
# DeepSeek (OpenAI 兼容)
API_PROVIDER=openai
OPENAI_API_KEY=your_key
MODEL=deepseek-chat

# 或 GPT-4
API_PROVIDER=openai
OPENAI_API_KEY=your_key
MODEL=gpt-4o

# 或 GLM/MiniMax (Anthropic 兼容)
API_PROVIDER=anthropic
ANTHROPIC_AUTH_TOKEN=your_key
ANTHROPIC_MODEL=MiniMax-M2.1
```

**无需修改代码！** UnifiedClient 会自动处理 API 调用格式和 logprobs 提取。

## 项目结构

```
concoll/
├── run_concoll.py              # 主入口
├── config.py                   # 配置管理
├── models/
│   ├── concoll_stage1.py      # Stage 1: Direct Prediction
│   ├── concoll_stage2.py       # Stage 2: RAG
│   └── concoll_stage3.py       # Stage 3: Multi-Agent
├── llm/
│   ├── unified_client.py       # 统一客户端
│   ├── anthropic_client.py     # Anthropic 兼容封装
│   └── gpt4o_client.py        # OpenAI 封装
├── data/
│   └── local_loader.py        # PrimeVul 加载器
├── evaluation/
│   └── metrics.py             # 评估指标
├── requirements.txt            # 依赖列表
├── results/                    # 结果输出
└── PrimeVul_v0.1/             # 数据集
    └── primevul_test_paired.jsonl
```

## 三阶段框架

### Stage 1: Direct Prediction
```
输入: 代码片段
↓
API 调用: 单个 LLM
↓
计算: C.S. = P(top-1) - P(top-2)
↓
判断: C.S. ≥ threshold? 且 top-1 为 Yes/No?
├─ YES → 接受预测（低成本）
└─ NO  → 进入 Stage 2
```

### Stage 2: RAG
```
输入: 低置信度代码
↓
检索: 从训练集找相似案例（50% 漏洞 + 50% 安全）
↓
API 调用: LLM with examples
↓
输出: 最终预测
```

### Stage 3: Multi-Agent
```
输入: 仍然不确定的代码
↓
并行调用: 3 个不同角色的 Agent
- Security Analyst
- Penetration Tester
- Software Security Engineer
↓
投票: 多数投票
↓
输出: 最终预测
```

**注意**: Stage 3 (多智能体协作) 默认启用。如需禁用，可以修改 `run_concoll.py` 中 `ConCollFramework` 的 `use_stage3` 参数。

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--max-samples N` | 最大样本数 | 全部 |
| `--confidence-threshold` | 置信度阈值 | 0.3 |
| `--test-data` | 使用合成测试数据 | - |
| `--simulate` | 模拟模式（固定分流比例） | - |
| `--stage1-ratio` | Stage 1 比例 | 0.7 |
| `--stage2-ratio` | Stage 2 比例 | 0.25 |
| `--stage3-ratio` | Stage 3 比例 | 0.05 |
| `--force-stages` | 强制所有样本通过所有阶段 | - |

## 结果格式

```json
{
  "method": "concoll",
  "num_samples": 100,
  "stage_stats": {
    "stage1_accepted": 70,
    "stage2_used": 25,
    "stage3_used": 5
  },
  "binary_metrics": {
    "accuracy": 0.72,
    "precision": 0.70,
    "recall": 0.73,
    "f1": 0.72
  }
}
```

## Citation

```bibtex
@inproceedings{tsai2025concoll,
  title={A Sequential Multi-Stage Approach for Code Vulnerability Detection via Confidence- and Collaboration-based Decision Making},
  author={Tsai, Chung-Nan and Wang, Xin and Lee, Cheng-Hsiung and Lin, Ching-Sheng},
  booktitle={EMNLP},
  year={2025}
}
```
