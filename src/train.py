import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from .dataset import create_dataloaders
from .model import create_model

class Trainer:
    """训练器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建数据加载器
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # 创建模型
        self.model = create_model(config).to(self.device)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs'],
            eta_min=1e-6
        )
        
        # 日志
        self.writer = SummaryWriter(config['paths']['log_dir'])
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["training"]["epochs"]}')
        
        for batch_idx, (data, labels) in enumerate(pbar):
            data, labels = data.to(self.device), labels.to(self.device)
            
            # 前向传播
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算指标
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        # 记录日志
        self.writer.add_scalar('Train/Loss', avg_loss, epoch)
        self.writer.add_scalar('Train/Accuracy', accuracy, epoch)
        
        return avg_loss, accuracy
    
    def validate(self, epoch: int) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in self.val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # 记录日志
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', accuracy, epoch)
        self.writer.add_scalar('Val/F1', f1, epoch)
        
        return avg_loss, accuracy, f1
    
    def test(self) -> dict:
        """测试"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                outputs = self.model(data)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'confusion_matrix': conf_matrix
        }
    
    def train(self):
        """训练循环"""
        print(f"使用设备: {self.device}")
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        print(f"验证样本数: {len(self.val_loader.dataset)}")
        print(f"测试样本数: {len(self.test_loader.dataset)}")
        print("-" * 50)
        
        for epoch in range(self.config['training']['epochs']):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc, val_f1 = self.validate(epoch)
            
            # 学习率调度
            self.scheduler.step()
            
            # 打印结果
            print(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            print("-" * 50)
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_model(os.path.join(self.config['paths']['model_dir'], 'best_model.pth'))
                print(f"保存最佳模型，验证准确率: {val_acc:.4f}")
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= self.config['training']['early_stopping']:
                print(f"早停，最佳验证准确率: {self.best_val_acc:.4f}")
                break
        
        # 测试
        print("开始测试...")
        test_results = self.test()
        print(f"测试准确率: {test_results['accuracy']:.4f}")
        print(f"测试F1分数: {test_results['f1']:.4f}")
        print("混淆矩阵:")
        print(test_results['confusion_matrix'])
        
        # 保存测试结果
        self.save_test_results(test_results)
        
        self.writer.close()
    
    def save_model(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_acc': self.best_val_acc
        }, path)
    
    def save_test_results(self, results: dict):
        """保存测试结果"""
        import pandas as pd
        
        # 保存混淆矩阵
        conf_matrix_df = pd.DataFrame(
            results['confusion_matrix'],
            index=self.config['model'].get('class_names', [f'Class {i}' for i in range(self.config['model']['num_classes'])]),
            columns=self.config['model'].get('class_names', [f'Class {i}' for i in range(self.config['model']['num_classes'])])
        )
        conf_matrix_df.to_csv(os.path.join(self.config['paths']['output_dir'], 'confusion_matrix.csv'))
        
        # 保存指标
        metrics = {
            'accuracy': results['accuracy'],
            'f1_score': results['f1']
        }
        pd.DataFrame([metrics]).to_csv(os.path.join(self.config['paths']['output_dir'], 'metrics.csv'), index=False)


def main():
    # 加载配置
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()