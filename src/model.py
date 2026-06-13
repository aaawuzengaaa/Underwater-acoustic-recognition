import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ChannelAttention(nn.Module):
    """通道注意力"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out.expand_as(x)


class CNNBlock(nn.Module):
    """CNN模块"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.attention = ChannelAttention(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.attention(x)
        x = self.pool(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerCNN(nn.Module):
    """Transformer-CNN混合模型"""
    
    def __init__(self, 
                 num_classes: int = 5,
                 input_dim: int = 128,
                 sequence_length: int = 100,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.d_model = d_model
        
        # CNN特征提取
        self.cnn = nn.Sequential(
            CNNBlock(1, 32, kernel_size=3, stride=1, padding=1),
            CNNBlock(32, 64, kernel_size=3, stride=1, padding=1),
            CNNBlock(64, 128, kernel_size=3, stride=1, padding=1),
        )
        
        # 计算CNN输出维度
        cnn_output_size = self._get_cnn_output_size(input_dim, sequence_length)
        
        # 特征映射
        self.feature_proj = nn.Linear(cnn_output_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer编码器
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def _get_cnn_output_size(self, input_dim: int, sequence_length: int) -> int:
        """计算CNN输出维度"""
        # 模拟输入
        x = torch.randn(1, 1, input_dim, sequence_length)
        x = self.cnn(x)
        return x.view(1, -1).size(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, 1, n_mels, sequence_length]
            
        Returns:
            输出张量 [batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # CNN特征提取
        x = self.cnn(x)  # [batch_size, 128, h, w]
        
        # 展平
        x = x.view(batch_size, -1)  # [batch_size, 128*h*w]
        
        # 特征映射
        x = self.feature_proj(x)  # [batch_size, d_model]
        
        # 添加序列维度
        x = x.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # 位置编码
        x = x.transpose(0, 1)  # [1, batch_size, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch_size, 1, d_model]
        
        # Transformer编码
        x = self.transformer_encoder(x)  # [batch_size, 1, d_model]
        
        # 取序列平均
        x = x.mean(dim=1)  # [batch_size, d_model]
        
        # 分类
        x = self.classifier(x)  # [batch_size, num_classes]
        
        return x


def create_model(config: dict) -> TransformerCNN:
    """创建模型"""
    model = TransformerCNN(
        num_classes=config['model']['num_classes'],
        input_dim=config['model']['input_dim'],
        sequence_length=config['model']['sequence_length'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )
    return model