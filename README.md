# ConColl: Sequential Multi-Stage Vulnerability Detection

复现 EMNLP 2025 论文 "A Sequential Multi-Stage Approach for Code Vulnerability Detection via Confidence- and Collaboration-based Decision Making"

## Overview

ConColl 通过基于置信度的顺序决策，动态选择三种检测策略：

| Stage | 方法 | 触发条件 | 占比 |
|-------|------|----------|------|
| **Stage 1** | Direct Prediction | Confidence Score ≥ threshold | 大部分样本 |
| **Stage 2** | RAG with examples | 低置信度 | 模糊案例 |
| **Stage 3** | Multi-Agent | Stage 2 仍不确定 | 困难案例 |

**置信度分数 (C.S.)** = P(top-1 token) - P(top-2 token)

- 高 C.S. → 接受直接预测（低成本）
- 低 C.S. → 进入 RAG 或多智能体（高精度）

## 实现状态

### ✅ 已完成

| 组件 | 状态 | 说明 |
|------|------|------|
| **三阶段框架** | ✓ | 完整实现 |
| **Stage 1** | ✓ | Direct Prediction + 置信度接口 |
| **Stage 2** | ✓ | RAG 检索增强生成 |
| **Stage 3** | ✓ | 多智能体协作 |
| **PrimeVul 支持** | ✓ | 本地 JSONL 加载 |
| **模拟模式** | ✓ | GLM 小样本测试 |
| **GPT 切换** | ✓ | 统一客户端，一键切换 |

### 模型支持

| 特性 | GLM/MiniMax | GPT-4 |
|------|-------------|-------|
| **logprobs 支持** | ✗ | ✓ |
| **置信度计算** | 低置信度 0.15（自动进入 Stage 2/3） | 真实计算 |
| **三阶段分流** | ✓ 自动 | ✓ 自动 |

## 快速开始

```bash
cd /data/lx/concoll
source .venv/bin/activate

# 使用真实 PrimeVul 数据集测试（推荐）
python run_concoll.py --max-samples 20

# 使用合成测试数据
python run_concoll.py --test-data

# 模拟模式（固定比例分流）
python run_concoll.py --test-data --simulate
```

## 切换到 GPT-4

### 配置步骤

修改 `.env` 文件：

```bash
# 当前配置 (GLM)
API_PROVIDER=anthropic
ANTHROPIC_AUTH_TOKEN=your_glm_key
ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic
ANTHROPIC_MODEL=glm-4.7

# 切换到 GPT-4
API_PROVIDER=openai
OPENAI_API_KEY=your_openai_key
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL=gpt-4
```

**无需修改代码！** UnifiedClient 会自动处理 API 调用格式和 logprobs 提取。

### 模型对比

| 特性 | GLM/MiniMax | GPT-4 |
|------|-------------|-------|
| logprobs | ✗ | ✓ |
| 置信度来源 | 低置信度 0.15 | 真实计算 |
| 三阶段分流 | ✓ 自动 | ✓ 自动 |
| 适用场景 | 测试/开发 | 生产使用 |

### 置信度计算原理

```python
# GPT-4 logprobs 格式 (真实)
logprobs = [
    {"token": "VULNERABLE", "logprob": -0.1},   # P ≈ 0.90
    {"token": "SAFE",      "logprob": -2.3}    # P ≈ 0.10
]
# 置信度 = 0.90 - 0.10 = 0.80

# GLM/MiniMax (无 logprobs)
# 使用低置信度 0.15，确保样本自动进入 Stage 2/3 进行更详细分析
# Stage 1: 置信度 < 0.3 → 进入 Stage 2
# Stage 2: 置信度 < 0.3 → 进入 Stage 3 (多智能体)
```

### MiniMax/GLM 测试流程

```bash
# 1. 使用合成数据快速测试（推荐先运行）
python run_concoll.py --test-data

# 2. 使用模拟模式测试分阶段逻辑
python run_concoll.py --test-data --simulate

# 3. 确认流程正确后，切换到 GPT-4 进行生产测试
```

## 项目结构

```
concoll/
├── run_concoll.py             # ConColl 主入口
├── config.py                  # 配置管理
├── models/
│   ├── concoll_stage1.py       # Stage 1: Direct Prediction
│   ├── concoll_stage2.py       # Stage 2: RAG
│   └── concoll_stage3.py       # Stage 3: Multi-Agent
├── llm/
│   ├── unified_client.py       # 统一客户端（支持 GPT/GLM）
│   ├── anthropic_client.py     # Anthropic 兼容封装
│   └── gpt4o_client.py         # OpenAI/GPT 封装
├── data/
│   └── local_loader.py        # PrimeVul 加载器
├── evaluation/
│   └── metrics.py             # 评估指标
├── requirements.txt            # 依赖列表
├── results/                    # 结果输出目录
└── PrimeVul_v0.1/             # 本地数据集
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
判断: C.S. ≥ threshold?
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
- Security Analyst: 安全分析视角
- Code Reviewer: 代码质量视角
- Attacker: 攻击者视角
↓
投票: 多数投票
↓
输出: 最终预测
```

## 命令行参数

```bash
python run_concoll.py [OPTIONS]

Options:
  --test-data                使用合成测试数据
  --max-samples N             最大样本数
  --confidence-threshold     Stage 1 置信度阈值 (默认 0.3)
  --output-dir DIR            结果输出目录
  --force-stages              强制所有样本通过所有阶段
  --simulate                  启用模拟模式 (固定比例分流)
  --stage1-ratio              Stage 1 比例 (默认 0.7)
  --stage2-ratio              Stage 2 比例 (默认 0.25)
  --stage3-ratio              Stage 3 比例 (默认 0.05)
```

**注意**: Stage 3 (多智能体协作) 默认启用。如需禁用，可以修改 `run_concoll.py` 中 `ConCollFramework` 的 `use_stage3` 参数。

## 使用示例

### GLM 小样本测试
```bash
# 默认比例 (70% / 25% / 5%)
python run_concoll.py --max-samples 20 --simulate

# 自定义比例
python run_concoll.py --max-samples 20 --simulate \
  --stage1-ratio 0.6 --stage2-ratio 0.3 --stage3-ratio 0.1
```

### GPT-4 生产使用
```bash
# 修改 .env 切换到 GPT-4 后运行
python run_concoll.py --max-samples 100
```

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
