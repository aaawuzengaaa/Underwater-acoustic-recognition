import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional

class UnderwaterAcousticDataset(Dataset):
    """水下声学信号数据集"""
    
    def __init__(self, 
                 data_dir: str,
                 sample_rate: int = 44100,
                 duration: float = 2.0,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 transform=None):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录
            sample_rate: 采样率
            duration: 音频时长（秒）
            n_mels: 梅尔滤波器组数量
            n_fft: FFT窗口大小
            hop_length: 帧移
            transform: 数据增强变换
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.transform = transform
        
        # 类别映射
        self.classes = ['normal', 'propeller', 'engine', 'pump', 'valve']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 加载文件列表
        self.file_list = self._load_file_list()
        
    def _load_file_list(self) -> List[Tuple[str, int]]:
        """加载文件列表"""
        file_list = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for filename in os.listdir(class_dir):
                if filename.endswith(('.wav', '.flac', '.mp3')):
                    filepath = os.path.join(class_dir, filename)
                    label = self.class_to_idx[class_name]
                    file_list.append((filepath, label))
        
        return file_list
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """获取单个样本"""
        filepath, label = self.file_list[idx]
        
        # 加载音频
        audio, sr = librosa.load(filepath, sr=self.sample_rate)
        
        # 裁剪或填充到固定长度
        target_length = int(self.sample_rate * self.duration)
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # 标准化
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        
        # 转换为tensor
        mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0)  # 添加通道维度
        
        # 数据增强
        if self.transform:
            mfcc_tensor = self.transform(mfcc_tensor)
        
        return mfcc_tensor, label


class SpecAugment:
    """SpecAugment数据增强"""
    
    def __init__(self, 
                 time_mask_num: int = 2,
                 freq_mask_num: int = 2,
                 time_mask_param: int = 10,
                 freq_mask_param: int = 15):
        self.time_mask_num = time_mask_num
        self.freq_mask_num = freq_mask_num
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """应用SpecAugment"""
        # 频率掩码
        for _ in range(self.freq_mask_num):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, spec.shape[2] - f)
            spec[:, :, f0:f0+f, :] = 0
        
        # 时间掩码
        for _ in range(self.time_mask_num):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, spec.shape[1] - t)
            spec[:, :, :, t0:t0+t] = 0
        
        return spec


def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器"""
    # 数据增强
    train_transform = SpecAugment(
        time_mask_num=config['augmentation']['time_mask_num'],
        freq_mask_num=config['augmentation']['freq_mask_num'],
        time_mask_param=config['augmentation']['time_mask_param'],
        freq_mask_param=config['augmentation']['freq_mask_param']
    )
    
    # 创建数据集
    train_dataset = UnderwaterAcousticDataset(
        data_dir=os.path.join(config['paths']['data_dir'], 'train'),
        sample_rate=config['data']['sample_rate'],
        duration=config['data']['duration'],
        n_mels=config['data']['n_mels'],
        n_fft=config['data']['n_fft'],
        hop_length=config['data']['hop_length'],
        transform=train_transform
    )
    
    val_dataset = UnderwaterAcousticDataset(
        data_dir=os.path.join(config['paths']['data_dir'], 'val'),
        sample_rate=config['data']['sample_rate'],
        duration=config['data']['duration'],
        n_mels=config['data']['n_mels'],
        n_fft=config['data']['n_fft'],
        hop_length=config['data']['hop_length']
    )
    
    test_dataset = UnderwaterAcousticDataset(
        data_dir=os.path.join(config['paths']['data_dir'], 'test'),
        sample_rate=config['data']['sample_rate'],
        duration=config['data']['duration'],
        n_mels=config['data']['n_mels'],
        n_fft=config['data']['n_fft'],
        hop_length=config['data']['hop_length']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader