import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from config import MODEL_CONFIG

class EyeStateClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(EyeStateClassifier, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        # 第三个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)
        
        # 第四个卷积块
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(0.25)
        
        # 全连接层
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.dropout6 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # 第一个卷积块
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        # 第二个卷积块
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        # 第三个卷积块
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        # 第四个卷积块
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout5(x)
        
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.dropout6(x)
        
        x = self.fc3(x)
        
        return x

class EyePredictor:
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model = EyeStateClassifier(MODEL_CONFIG['num_classes'])
        self.model.to(self.device)
        
        if model_path and torch.cuda.is_available():
            self.load_model(model_path)
        
        self.transform = transforms.Compose([
            transforms.Resize(MODEL_CONFIG['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.model.eval()
    
    def load_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"模型加载成功: {model_path}")
        except Exception as e:
            print(f"警告: 模型加载失败: {e}")
    
    def preprocess_eye_region(self, eye_region):
        if eye_region is None or eye_region.size == 0:
            return None
        
        # 确保图像大小合适
        if eye_region.shape[0] < 20 or eye_region.shape[1] < 20:
            return None
        
        # 转换为PIL图像
        if len(eye_region.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(eye_region)
        
        # 应用预处理
        tensor = self.transform(pil_image).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict_eye_state(self, eye_region):
        tensor = self.preprocess_eye_region(eye_region)
        if tensor is None:
            return None, 0.0
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return prediction, confidence
    
    def predict_both_eyes(self, left_eye, right_eye):
        left_pred, left_conf = self.predict_eye_state(left_eye)
        right_pred, right_conf = self.predict_eye_state(right_eye)
        
        results = {
            'left_eye': {'prediction': left_pred, 'confidence': left_conf},
            'right_eye': {'prediction': right_pred, 'confidence': right_conf}
        }
        
        # 综合判断：如果任一眼睛闭合，则判断为闭眼
        if left_pred is not None and right_pred is not None:
            combined_prediction = 0 if (left_pred == 0 or right_pred == 0) else 1
            combined_confidence = (left_conf + right_conf) / 2
            results['combined'] = {
                'prediction': combined_prediction, 
                'confidence': combined_confidence
            }
        
        return results

# 简化的预训练模型创建函数（如果没有训练数据可以使用基础模型）
def create_simple_predictor():
    """创建一个基于经验规则的简单预测器，用于演示"""
    class SimplePredictior:
        def predict_both_eyes(self, left_eye, right_eye):
            # 基于图像亮度的简单规则
            def analyze_brightness(eye_region):
                if eye_region is None or eye_region.size == 0:
                    return None, 0.0
                
                gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if len(eye_region.shape) == 3 else eye_region
                mean_brightness = np.mean(gray)
                
                # 简单规则：亮度低于阈值认为是闭眼
                threshold = 50
                prediction = 0 if mean_brightness < threshold else 1
                confidence = abs(mean_brightness - threshold) / 255
                
                return prediction, confidence
            
            left_pred, left_conf = analyze_brightness(left_eye)
            right_pred, right_conf = analyze_brightness(right_eye)
            
            results = {
                'left_eye': {'prediction': left_pred, 'confidence': left_conf},
                'right_eye': {'prediction': right_pred, 'confidence': right_conf}
            }
            
            if left_pred is not None and right_pred is not None:
                combined_prediction = 0 if (left_pred == 0 or right_pred == 0) else 1
                combined_confidence = (left_conf + right_conf) / 2
                results['combined'] = {
                    'prediction': combined_prediction,
                    'confidence': combined_confidence
                }
            
            return results
    
    return SimplePredictior()