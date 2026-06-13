import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional
import pyaudio
import threading
import queue

class AudioProcessor:
    """音频处理器"""
    
    def __init__(self, sample_rate: int = 44100, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        加载音频文件
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            (音频数据, 采样率)
        """
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio, sr
    
    def save_audio(self, audio: np.ndarray, file_path: str, sr: Optional[int] = None):
        """
        保存音频文件
        
        Args:
            audio: 音频数据
            file_path: 输出文件路径
            sr: 采样率
        """
        if sr is None:
            sr = self.sample_rate
        sf.write(file_path, audio, sr)
    
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """归一化音频"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    def remove_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """移除静音部分"""
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return audio_trimmed
    
    def add_noise(self, audio: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
        """添加噪声"""
        noise = np.random.randn(len(audio)) * noise_level
        return audio + noise
    
    def time_stretch(self, audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
        """时间拉伸"""
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio: np.ndarray, n_steps: float = 0.0) -> np.ndarray:
        """音调偏移"""
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
    
    def extract_features(self, audio: np.ndarray, sr: int, feature_type: str = 'mfcc') -> np.ndarray:
        """
        提取音频特征
        
        Args:
            audio: 音频数据
            sr: 采样率
            feature_type: 特征类型 ('mfcc', 'spectral', 'chroma')
            
        Returns:
            特征数组
        """
        if feature_type == 'mfcc':
            return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        elif feature_type == 'spectral':
            return librosa.feature.spectral_centroid(y=audio, sr=sr)
        elif feature_type == 'chroma':
            return librosa.feature.chroma_stft(y=audio, sr=sr)
        else:
            raise ValueError(f"不支持的特征类型: {feature_type}")
    
    def compute_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """计算频谱图"""
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
        return librosa.power_to_db(spectrogram, ref=np.max)


class RealtimeAudioProcessor:
    """实时音频处理器"""
    
    def __init__(self, sample_rate: int = 44100, chunk_size: int = 1024, buffer_size: int = 10):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        
        # 音频流
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # 缓冲区
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.is_running = False
        
        # 处理线程
        self.process_thread = None
        
    def start(self, callback):
        """
        开始实时音频处理
        
        Args:
            callback: 回调函数，接收音频块作为参数
        """
        self.is_running = True
        
        # 打开音频流
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        # 启动处理线程
        self.process_thread = threading.Thread(target=self._process_audio, args=(callback,))
        self.process_thread.start()
        
        print("实时音频处理已启动")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频流回调"""
        if not self.buffer.full():
            self.buffer.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def _process_audio(self, callback):
        """处理音频线程"""
        while self.is_running:
            try:
                # 获取音频块
                audio_chunk = self.buffer.get(timeout=0.1)
                
                # 转换为numpy数组
                audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
                
                # 调用回调函数
                callback(audio_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理音频时出错: {e}")
    
    def stop(self):
        """停止实时音频处理"""
        self.is_running = False
        
        if self.process_thread:
            self.process_thread.join()
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.audio.terminate()
        print("实时音频处理已停止")


def audio_to_spectrogram(audio: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    将音频转换为频谱图
    
    Args:
        audio: 音频数据
        sr: 采样率
        n_fft: FFT窗口大小
        hop_length: 帧移
        
    Returns:
        频谱图数组
    """
    spectrogram = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length
    )
    return librosa.power_to_db(spectrogram, ref=np.max)


def spectrogram_to_audio(spectrogram: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    将频谱图转换为音频
    
    Args:
        spectrogram: 频谱图数组
        sr: 采样率
        n_fft: FFT窗口大小
        hop_length: 帧移
        
    Returns:
        音频数据
    """
    spectrogram_db = librosa.db_to_power(spectrogram)
    audio = librosa.feature.inverse.mel_to_audio(
        spectrogram_db, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length
    )
    return audio