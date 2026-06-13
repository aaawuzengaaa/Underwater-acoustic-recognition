import os
import yaml
import torch
import librosa
import numpy as np
from typing import Dict, List, Tuple
import soundfile as sf

from .model import create_model

class Predictor:
    """预测器"""
    
    def __init__(self, model_path: str, config_path: str = 'configs/config.yaml'):
        """
        初始化预测器
        
        Args:
            model_path: 模型权重路径
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 类别名称
        self.class_names = ['正常', '螺旋桨', '发动机', '水泵', '阀门']
        
        # 创建模型
        self.model = create_model(self.config).to(self.device)
        
        # 加载权重
        self.load_model(model_path)
        
        # 模型参数
        self.sample_rate = self.config['data']['sample_rate']
        self.duration = self.config['data']['duration']
        self.n_mels = self.config['data']['n_mels']
        self.n_fft = self.config['data']['n_fft']
        self.hop_length = self.config['data']['hop_length']
    
    def load_model(self, model_path: str):
        """加载模型权重"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"模型加载成功: {model_path}")
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        预处理音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            预处理后的特征张量
        """
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
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
        mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).unsqueeze(0)  # [1, 1, n_mels, seq_len]
        
        return mfcc_tensor
    
    def predict(self, audio_path: str) -> Dict[str, float]:
        """
        预测单个音频
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            预测结果字典 {类别名: 概率}
        """
        # 预处理
        input_tensor = self.preprocess_audio(audio_path).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
        
        # 转换为字典
        results = {}
        for i, prob in enumerate(probs[0].cpu().numpy()):
            results[self.class_names[i]] = float(prob)
        
        return results
    
    def predict_with_confidence(self, audio_path: str, threshold: float = 0.5) -> Tuple[str, float, Dict[str, float]]:
        """
        带置信度的预测
        
        Args:
            audio_path: 音频文件路径
            threshold: 置信度阈值
            
        Returns:
            (预测类别, 置信度, 所有类别概率)
        """
        results = self.predict(audio_path)
        
        # 获取最高概率类别
        max_class = max(results, key=results.get)
        max_prob = results[max_class]
        
        # 判断是否达到阈值
        if max_prob >= threshold:
            return max_class, max_prob, results
        else:
            return "未知", max_prob, results
    
    def predict_batch(self, audio_paths: List[str]) -> List[Dict[str, float]]:
        """
        批量预测
        
        Args:
            audio_paths: 音频文件路径列表
            
        Returns:
            预测结果列表
        """
        results = []
        for audio_path in audio_paths:
            result = self.predict(audio_path)
            results.append(result)
        return results
    
    def predict_realtime(self, audio_chunk: np.ndarray, sr: int) -> Dict[str, float]:
        """
        实时预测（处理音频块）
        
        Args:
            audio_chunk: 音频数据块
            sr: 采样率
            
        Returns:
            预测结果字典
        """
        # 调整采样率
        if sr != self.sample_rate:
            audio_chunk = librosa.resample(audio_chunk, orig_sr=sr, target_sr=self.sample_rate)
        
        # 裁剪或填充
        target_length = int(self.sample_rate * self.duration)
        if len(audio_chunk) > target_length:
            audio_chunk = audio_chunk[:target_length]
        else:
            audio_chunk = np.pad(audio_chunk, (0, target_length - len(audio_chunk)))
        
        # 提取特征
        mfcc = librosa.feature.mfcc(
            y=audio_chunk, 
            sr=self.sample_rate, 
            n_mfcc=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # 标准化
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        
        # 转换为tensor
        mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(mfcc_tensor)
            probs = torch.softmax(outputs, dim=1)
        
        # 转换为字典
        results = {}
        for i, prob in enumerate(probs[0].cpu().numpy()):
            results[self.class_names[i]] = float(prob)
        
        return results


def main():
    """命令行预测"""
    import argparse
    
    parser = argparse.ArgumentParser(description='水下声学信号预测')
    parser.add_argument('--audio', type=str, required=True, help='音频文件路径')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='模型权重路径')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    parser.add_argument('--threshold', type=float, default=0.5, help='置信度阈值')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = Predictor(args.model, args.config)
    
    # 预测
    pred_class, confidence, all_probs = predictor.predict_with_confidence(args.audio, args.threshold)
    
    # 打印结果
    print(f"\n预测结果:")
    print(f"  预测类别: {pred_class}")
    print(f"  置信度: {confidence:.4f}")
    print(f"\n所有类别概率:")
    for cls, prob in all_probs.items():
        print(f"  {cls}: {prob:.4f}")


if __name__ == '__main__':
    main()