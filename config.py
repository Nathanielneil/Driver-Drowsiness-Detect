import os

# 模型配置
MODEL_CONFIG = {
    'input_size': (224, 224),
    'num_classes': 2,  # 0: closed, 1: open
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50
}

# 疲劳检测配置
DROWSINESS_CONFIG = {
    'ear_threshold': 0.25,  # Eye Aspect Ratio阈值
    'consecutive_frames': 20,  # 连续帧数判断疲劳
    'alarm_duration': 1.0  # 警报持续时间
}

# 文件路径配置
PATHS = {
    'models_dir': 'models',
    'data_dir': 'data',
    'logs_dir': 'logs',
    'output_dir': 'output'
}

# dlib人脸关键点检测器配置
DLIB_CONFIG = {
    'predictor_path': 'models/shape_predictor_68_face_landmarks.dat'
}

# 眼部关键点索引 (68点人脸标记)
EYE_LANDMARKS = {
    'left_eye': list(range(42, 48)),
    'right_eye': list(range(36, 42))
}

# 创建必要目录
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)