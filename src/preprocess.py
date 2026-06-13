import os
import yaml
import librosa
import numpy as np
from typing import List, Tuple
import soundfile as sf
from pathlib import Path
import argparse

class AudioPreprocessor:
    """音频预处理器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.sample_rate = config['data']['sample_rate']
        self.duration = config['data']['duration']
        self.target_length = int(self.sample_rate * self.duration)
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """加载音频文件"""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio, sr
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """归一化音频"""
        # 峰值归一化
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def trim_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """裁剪静音部分"""
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return audio_trimmed
    
    def pad_or_crop(self, audio: np.ndarray) -> np.ndarray:
        """填充或裁剪到固定长度"""
        if len(audio) > self.target_length:
            # 裁剪
            audio = audio[:self.target_length]
        else:
            # 填充
            audio = np.pad(audio, (0, self.target_length - len(audio)))
        return audio
    
    def add_noise(self, audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.randn(len(audio)) * noise_factor
        audio_noisy = audio + noise
        return audio_noisy
    
    def time_stretch(self, audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
        """时间拉伸"""
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio: np.ndarray, sr: int, n_steps: float = 0.0) -> np.ndarray:
        """音调偏移"""
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    def extract_mfcc(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """提取MFCC特征"""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.config['data']['n_mels'],
            n_fft=self.config['data']['n_fft'],
            hop_length=self.config['data']['hop_length']
        )
        return mfcc
    
    def process_audio(self, audio_path: str, augment: bool = False) -> np.ndarray:
        """
        处理单个音频文件
        
        Args:
            audio_path: 音频文件路径
            augment: 是否进行数据增强
            
        Returns:
            处理后的MFCC特征
        """
        # 加载音频
        audio, sr = self.load_audio(audio_path)
        
        # 预处理
        audio = self.normalize_audio(audio)
        audio = self.trim_silence(audio)
        audio = self.pad_or_crop(audio)
        
        # 数据增强
        if augment:
            # 随机选择增强方式
            augment_type = np.random.choice(['noise', 'stretch', 'pitch', 'none'])
            
            if augment_type == 'noise':
                audio = self.add_noise(audio, self.config['augmentation']['noise_factor'])
            elif augment_type == 'stretch':
                rate = np.random.uniform(0.8, 1.2)
                audio = self.time_stretch(audio, rate)
                audio = self.pad_or_crop(audio)  # 重新填充
            elif augment_type == 'pitch':
                n_steps = np.random.uniform(-2, 2)
                audio = self.pitch_shift(audio, sr, n_steps)
        
        # 提取特征
        mfcc = self.extract_mfcc(audio, sr)
        
        # 标准化
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        
        return mfcc
    
    def process_directory(self, input_dir: str, output_dir: str, augment: bool = False):
        """
        处理整个目录
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            augment: 是否进行数据增强
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理的文件数量
        processed_count = 0
        
        # 遍历输入目录
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(('.wav', '.flac', '.mp3')):
                    # 输入文件路径
                    input_path = os.path.join(root, file)
                    
                    # 计算相对路径
                    rel_path = os.path.relpath(root, input_dir)
                    
                    # 创建输出子目录
                    output_subdir = os.path.join(output_dir, rel_path)
                    os.makedirs(output_subdir, exist_ok=True)
                    
                    # 输出文件路径
                    output_file = os.path.splitext(file)[0] + '.npy'
                    output_path = os.path.join(output_subdir, output_file)
                    
                    try:
                        # 处理音频
                        mfcc = self.process_audio(input_path, augment)
                        
                        # 保存特征
                        np.save(output_path, mfcc)
                        
                        processed_count += 1
                        print(f"处理完成: {input_path} -> {output_path}")
                        
                    except Exception as e:
                        print(f"处理失败: {input_path}, 错误: {e}")
        
        print(f"\n处理完成! 共处理 {processed_count} 个文件")


def split_dataset(input_dir: str, output_dir: str, config: dict):
    """
    划分数据集为训练集、验证集、测试集
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        config: 配置
    """
    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 获取所有类别
    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for cls in classes:
        cls_dir = os.path.join(input_dir, cls)
        files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
        
        # 打乱文件列表
        np.random.shuffle(files)
        
        # 计算划分点
        n_files = len(files)
        n_train = int(n_files * config['data']['train_split'])
        n_val = int(n_files * config['data']['val_split'])
        
        # 划分
        train_files = files[:n_train]
        val_files = files[n_train:n_train+n_val]
        test_files = files[n_train+n_val:]
        
        # 创建子目录
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
        
        # 移动文件
        for f in train_files:
            src = os.path.join(cls_dir, f)
            dst = os.path.join(train_dir, cls, f)
            os.rename(src, dst)
        
        for f in val_files:
            src = os.path.join(cls_dir, f)
            dst = os.path.join(val_dir, cls, f)
            os.rename(src, dst)
        
        for f in test_files:
            src = os.path.join(cls_dir, f)
            dst = os.path.join(test_dir, cls, f)
            os.rename(src, dst)
        
        print(f"类别 {cls}: 训练集 {len(train_files)}, 验证集 {len(val_files)}, 测试集 {len(test_files)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='音频数据预处理')
    parser.add_argument('--input', type=str, required=True, help='输入目录')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件')
    parser.add_argument('--augment', action='store_true', help='是否进行数据增强')
    parser.add_argument('--split', action='store_true', help='是否划分数据集')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建预处理器
    preprocessor = AudioPreprocessor(config)
    
    # 处理目录
    temp_dir = os.path.join(args.output, 'temp_features')
    preprocessor.process_directory(args.input, temp_dir, args.augment)
    
    # 划分数据集
    if args.split:
        split_dataset(temp_dir, args.output, config)
        # 删除临时目录
        import shutil
        shutil.rmtree(temp_dir)
    
    print("预处理完成!")


if __name__ == '__main__':
    main()