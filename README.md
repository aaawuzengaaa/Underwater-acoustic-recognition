# 水下声学信号智能识别系统

基于深度学习的水下声学信号智能识别系统，支持水下目标检测、分类与跟踪。

## 项目简介

本项目针对水下目标识别中背景噪声复杂、目标特征相似度高的问题，构建了一套完整的水下声学信号智能识别系统。系统采用Transformer与CNN混合架构，结合数据增强和迁移学习技术，在公开数据集上识别准确率达94.5%。

## 主要功能

- 水下声学信号预处理（去噪、特征提取）
- 基于深度学习的目标分类识别
- 实时音频流处理与识别
- 识别结果可视化展示
- 模型训练与评估

## 技术架构

```
音频输入 → 预处理 → 特征提取 → 深度学习模型 → 分类结果 → 可视化
```

## 项目结构

```
underwater-acoustic-recognition/
├── README.md              # 项目说明文档
├── requirements.txt       # 依赖包列表
├── setup.py              # 项目安装配置
├── configs/              # 配置文件
│   └── config.yaml       # 模型配置
├── src/                  # 源代码
│   ├── __init__.py
│   ├── dataset.py        # 数据集处理
│   ├── model.py          # 模型定义
│   ├── train.py          # 训练脚本
│   ├── predict.py        # 预测脚本
│   └── preprocess.py     # 数据预处理
├── utils/                # 工具函数
│   ├── __init__.py
│   ├── audio_utils.py    # 音频处理工具
│   └── visualization.py  # 可视化工具
├── data/                 # 数据目录
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后数据
├── models/               # 模型权重
├── notebooks/            # Jupyter笔记本
│   └── exploration.ipynb # 数据探索
└── results/              # 结果输出
```

## 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- torchaudio >= 2.0

## 安装

1. 克隆项目
```bash
git clone https://github.com/zhangzeng-ai/underwater-acoustic-recognition.git
cd underwater-acoustic-recognition
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

## 数据准备

1. 将音频数据放入 `data/raw/` 目录
2. 运行数据预处理脚本：
```bash
python -m src.preprocess --input data/raw --output data/processed
```

## 模型训练

```bash
python -m src.train --config configs/config.yaml
```

## 模型预测

```bash
python -m src.predict --audio path/to/audio.wav --model models/best_model.pth
```

## 模型性能

| 指标 | 数值 |
|------|------|
| 准确率 | 94.5% |
| F1-Score | 0.94 |
| 单次推理时间 | 15ms |

## 技术细节

### 模型架构
- 基础网络：Transformer + CNN混合架构
- 特征提取：MFCC + FBank + 时频图
- 注意力机制：自注意力 + 通道注意力

### 数据增强
- 时频图变换（旋转、缩放、裁剪）
- 噪声注入（高斯噪声、环境噪声）
- 时间扭曲（拉伸、压缩）
- SpecAugment

### 迁移学习
- 预训练数据集：ESC-50、UrbanSound8K
- 微调策略：冻结底层，微调分类头

## 许可证

MIT License

## 联系方式

- Email: 739155644@qq.com
- GitHub: https://github.com/zhangzeng-ai