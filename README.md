# 驾驶员疲劳检测系统

基于PyTorch的实时驾驶员疲劳检测系统，使用计算机视觉技术监测驾驶员眼部状态，实时判断疲劳程度并发出警报。

## 功能特性

- 🎯 **实时人脸检测**: 基于dlib的高精度人脸检测
- 👁️ **眼部状态分析**: 支持EAR算法和CNN深度学习模型
- ⚡ **多重检测机制**: 结合眼部纵横比(EAR)和深度学习预测
- 🚨 **智能警报系统**: 音频警报和视觉提示
- 📊 **实时统计**: FPS、检测精度等性能指标
- 🔧 **灵活配置**: 可调节检测阈值和参数

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   视频输入       │    │   人脸检测       │    │   眼部定位       │
│  (摄像头/文件)   │--->│  (dlib检测器)   │--->│ (68点关键点)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   疲劳判断       │<---│   状态分析       │<---│   眼部区域提取   │
│  (综合决策)      │    │ (EAR + CNN)     │    │  (ROI提取)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │
        v
┌─────────────────┐
│   警报输出       │
│ (音频+视觉提示)  │
└─────────────────┘
```

## 安装部署

### 1. 环境要求

- Python 3.7+
- 摄像头 (USB/内置)
- 支持的操作系统: Linux, Windows, macOS

### 2. 快速安装

```bash
# 克隆项目到服务器
git clone <project_url> driver_drowsiness_detection
cd driver_drowsiness_detection

# 运行自动安装脚本
python setup.py
```

### 3. 手动安装

```bash
# 安装Python依赖
pip install -r requirements.txt

# 下载dlib人脸关键点检测器
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
mkdir -p models
mv shape_predictor_68_face_landmarks.dat models/

# 测试系统
python test_system.py
```


## 使用说明

### 基本使用

```bash
# 使用默认摄像头开始检测
python run_demo.py

# 使用指定摄像头
python run_demo.py --source 1

# 处理视频文件
python run_demo.py --source /path/to/video.mp4

# 保存结果视频
python run_demo.py --output /path/to/output.avi
```

### 高级配置

```bash
# 调整EAR阈值 (默认0.25)
python run_demo.py --ear-threshold 0.3

# 调整连续帧阈值 (默认20帧)
python run_demo.py --consecutive-frames 15

# 使用CNN模型 (需要训练好的模型)
python run_demo.py --model models/eye_classifier.pth
```

### 键盘控制

- `q`: 退出程序
- `s`: 显示实时统计信息

## 配置参数

编辑 `config.py` 文件调整系统参数:

```python
# 疲劳检测配置
DROWSINESS_CONFIG = {
    'ear_threshold': 0.25,        # EAR阈值，越小越敏感
    'consecutive_frames': 20,     # 连续闭眼帧数
    'alarm_duration': 1.0         # 警报持续时间(秒)
}

# 模型配置
MODEL_CONFIG = {
    'input_size': (224, 224),     # 输入图像尺寸
    'batch_size': 32,             # 批处理大小
    'learning_rate': 0.001        # 学习率
}
```

## API接口

### 核心类

#### DrowsinessDetector
```python
from drowsiness_detector import DrowsinessDetector

detector = DrowsinessDetector(use_cnn=False, model_path=None)
results, annotated_frame = detector.analyze_frame(frame)
```

#### FaceDetector  
```python
from face_detection import FaceDetector

face_detector = FaceDetector()
faces, gray = face_detector.detect_faces(frame)
landmarks = face_detector.get_landmarks(gray, face)
```

#### EyePredictor
```python
from eye_classifier import EyePredictor

predictor = EyePredictor(model_path="models/eye_model.pth")
prediction, confidence = predictor.predict_eye_state(eye_region)
```

## 性能优化

### 1. 硬件优化
- 使用GPU加速 (CUDA支持)
- 高性能摄像头 (>=30FPS)
- 充足内存 (推荐8GB+)

### 2. 软件优化
```python
# 降低输入分辨率提升速度
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 跳帧处理 (每N帧处理一次)
if frame_count % skip_frames == 0:
    results = detector.analyze_frame(frame)
```

### 3. 模型优化
- 使用TensorRT加速推理
- 模型量化 (FP16/INT8)
- 模型蒸馏压缩

## 项目结构

```
driver_drowsiness_detection/
├── config.py                    # 配置文件
├── face_detection.py           # 人脸检测模块  
├── eye_classifier.py           # 眼部分类器
├── drowsiness_detector.py      # 主检测器(GUI版本)
├── server_inference.py         # 服务器推理引擎
├── run_demo.py                 # 演示程序
├── setup.py                    # 安装脚本
├── deploy.sh                   # 服务器部署脚本
├── test_system.py              # 系统测试
├── test_image_inference.py     # 图片测试脚本
├── requirements.txt            # 依赖列表
├── README.md                   # 详细文档
├── 快速开始.md                 # 快速部署指南
├── demo/                       # 测试图片目录
│   ├── README.md              # 测试图片说明
│   └── .gitkeep               # 保持目录存在
├── models/                     # 模型文件
│   └── shape_predictor_68_face_landmarks.dat
├── data/                       # 数据目录
├── logs/                       # 日志目录
└── output/                     # 输出目录
```

## 技术原理

### 1. 眼部纵横比 (EAR)
```
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
```
- 正常睁眼: EAR > 0.25  
- 闭眼状态: EAR < 0.25

### 2. CNN分类模型
- 输入: 224x224 RGB眼部图像
- 架构: 4层卷积 + 3层全连接
- 输出: 2类概率 (睁眼/闭眼)

### 3. 疲劳判断逻辑
```python
if (EAR < threshold OR CNN_prediction == "closed") and consecutive_frames > limit:
    trigger_drowsiness_alert()
```

## 常见问题

### Q: 摄像头无法访问？
A: 检查摄像头连接，确保没有被其他程序占用，在Linux下可能需要权限:
```bash
sudo usermod -a -G video $USER
```

### Q: dlib安装失败？
A: 可能需要编译工具:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake

# CentOS/RHEL  
sudo yum install gcc-c++ cmake
```

### Q: 检测精度不够？
A: 调整参数或训练自定义模型:
- 降低EAR阈值增加敏感度
- 减少连续帧数要求
- 使用更大训练数据集

### Q: 性能较慢？
A: 优化建议:
- 降低视频分辨率
- 启用GPU加速  
- 使用更简单的模型

## 扩展功能

### 1. 多人检测
```python
# 处理多个人脸
for face in faces:
    landmarks = detector.get_landmarks(gray, face)
    # 为每个人脸独立检测
```

### 2. 远程监控
```python
# WebSocket实时传输
import websocket
# HTTP API接口
from flask import Flask
```

### 3. 数据记录
```python  
# CSV格式记录
import pandas as pd
df.to_csv('drowsiness_log.csv')
```

## 贡献指南

1. Fork项目仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)  
5. 创建Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

- 项目地址: [GitHub Repository]
- 问题反馈: [GitHub Issues]
- 邮箱: [your-email@example.com]

## 更新日志

### v1.0.0 (2024-01-xx)
- ✨ 初始版本发布
- 🎯 基础疲劳检测功能
- 📱 实时视频处理
- 🚨 音频警报系统

### 计划功能
- [ ] Web界面管理
- [ ] 移动端应用  
- [ ] 云端部署支持
- [ ] 更多深度学习模型
- [ ] 疲劳程度等级评估