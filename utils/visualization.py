import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import librosa
import librosa.display

class Visualizer:
    """可视化工具"""
    
    def __init__(self, figsize: tuple = (10, 6)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_waveform(self, audio: np.ndarray, sr: int, title: str = "Waveform"):
        """
        绘制波形图
        
        Args:
            audio: 音频数据
            sr: 采样率
            title: 图表标题
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        librosa.display.waveshow(audio, sr=sr, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        plt.tight_layout()
        return fig
    
    def plot_spectrogram(self, audio: np.ndarray, sr: int, title: str = "Spectrogram"):
        """
        绘制频谱图
        
        Args:
            audio: 音频数据
            sr: 采样率
            title: 图表标题
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        img = librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title(title)
        plt.tight_layout()
        return fig
    
    def plot_mfcc(self, audio: np.ndarray, sr: int, title: str = "MFCC"):
        """
        绘制MFCC特征
        
        Args:
            audio: 音频数据
            sr: 采样率
            title: 图表标题
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        img = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax)
        fig.colorbar(img, ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        return fig
    
    def plot_prediction_results(self, predictions: Dict[str, float], true_label: Optional[str] = None):
        """
        绘制预测结果
        
        Args:
            predictions: 预测结果字典 {类别名: 概率}
            true_label: 真实标签
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        classes = list(predictions.keys())
        probs = list(predictions.values())
        
        colors = ['green' if cls == true_label else 'blue' for cls in classes]
        
        bars = ax.bar(classes, probs, color=colors)
        ax.set_xlabel('Class')
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Results')
        ax.set_ylim([0, 1])
        
        # 添加数值标签
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, class_names: List[str]):
        """
        绘制混淆矩阵
        
        Args:
            confusion_matrix: 混淆矩阵
            class_names: 类别名称列表
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(
            confusion_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        return fig
    
    def plot_training_history(self, history: Dict[str, List[float]]):
        """
        绘制训练历史
        
        Args:
            history: 训练历史字典 {'train_loss': [...], 'val_loss': [...], ...}
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        if 'train_loss' in history and 'val_loss' in history:
            axes[0].plot(history['train_loss'], label='Train Loss')
            axes[0].plot(history['val_loss'], label='Val Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Loss Curve')
            axes[0].legend()
        
        # 准确率曲线
        if 'train_acc' in history and 'val_acc' in history:
            axes[1].plot(history['train_acc'], label='Train Accuracy')
            axes[1].plot(history['val_acc'], label='Val Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Accuracy Curve')
            axes[1].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_names: List[str], importance: np.ndarray):
        """
        绘制特征重要性
        
        Args:
            feature_names: 特征名称列表
            importance: 特征重要性数组
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 排序
        indices = np.argsort(importance)[::-1]
        
        # 绘制
        ax.bar(range(len(importance)), importance[indices])
        ax.set_xticks(range(len(importance)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Importance')
        ax.set_title('Feature Importance')
        
        plt.tight_layout()
        return fig


def save_figure(fig, path: str, dpi: int = 300):
    """
    保存图表
    
    Args:
        fig: matplotlib图表对象
        path: 保存路径
        dpi: 分辨率
    """
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"图表已保存: {path}")