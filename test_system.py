#!/usr/bin/env python3
"""
系统测试脚本
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path

def test_opencv():
    """测试OpenCV"""
    print("测试OpenCV...")
    try:
        import cv2
        print(f"✓ OpenCV版本: {cv2.__version__}")
        
        # 测试基本功能
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(img, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        print("✓ OpenCV基本功能正常")
        return True
    except Exception as e:
        print(f"❌ OpenCV测试失败: {e}")
        return False

def test_pytorch():
    """测试PyTorch"""
    print("测试PyTorch...")
    try:
        import torch
        import torchvision
        
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ TorchVision版本: {torchvision.__version__}")
        
        # 测试CUDA可用性
        if torch.cuda.is_available():
            print(f"✓ CUDA可用: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ CUDA不可用，将使用CPU")
        
        # 测试张量操作
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = x + y
        print("✓ PyTorch张量操作正常")
        return True
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")
        return False

def test_dlib():
    """测试dlib"""
    print("测试dlib...")
    try:
        import dlib
        print(f"✓ dlib版本: {dlib.DLIB_VERSION}")
        
        # 检查预测器文件
        predictor_path = Path("models/shape_predictor_68_face_landmarks.dat")
        if predictor_path.exists():
            print("✓ 人脸关键点预测器文件存在")
            
            # 测试加载
            try:
                predictor = dlib.shape_predictor(str(predictor_path))
                print("✓ 预测器加载成功")
            except Exception as e:
                print(f"⚠️  预测器加载失败: {e}")
        else:
            print("❌ 人脸关键点预测器文件不存在")
            print("   请运行: python setup.py")
            return False
        
        return True
    except Exception as e:
        print(f"❌ dlib测试失败: {e}")
        return False

def test_camera():
    """测试摄像头"""
    print("测试摄像头...")
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ 无法打开摄像头")
            return False
        
        # 测试读取帧
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"✓ 摄像头工作正常 (分辨率: {w}x{h})")
            
            # 保存测试图像
            test_img_path = Path("output/camera_test.jpg")
            cv2.imwrite(str(test_img_path), frame)
            print(f"✓ 测试图像已保存: {test_img_path}")
        else:
            print("❌ 无法读取摄像头帧")
            return False
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ 摄像头测试失败: {e}")
        return False

def test_face_detection():
    """测试人脸检测模块"""
    print("测试人脸检测模块...")
    try:
        from face_detection import FaceDetector
        
        detector = FaceDetector()
        print("✓ 人脸检测器初始化成功")
        
        # 创建测试图像 (简单的面部轮廓)
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(test_img, (320, 240), 80, (255, 255, 255), -1)  # 脸
        cv2.circle(test_img, (300, 220), 10, (0, 0, 0), -1)  # 左眼
        cv2.circle(test_img, (340, 220), 10, (0, 0, 0), -1)  # 右眼
        
        faces, gray = detector.detect_faces(test_img)
        print(f"✓ 人脸检测测试完成 (检测到 {len(faces)} 个人脸)")
        
        return True
        
    except Exception as e:
        print(f"❌ 人脸检测测试失败: {e}")
        return False

def test_eye_classifier():
    """测试眼部分类器"""
    print("测试眼部分类器...")
    try:
        from eye_classifier import create_simple_predictor
        
        predictor = create_simple_predictor()
        print("✓ 简单预测器创建成功")
        
        # 创建测试眼部图像
        open_eye = np.ones((50, 80, 3), dtype=np.uint8) * 200  # 明亮 = 睁眼
        closed_eye = np.ones((50, 80, 3), dtype=np.uint8) * 30  # 暗 = 闭眼
        
        results_open = predictor.predict_both_eyes(open_eye, open_eye)
        results_closed = predictor.predict_both_eyes(closed_eye, closed_eye)
        
        print(f"✓ 睁眼预测: {results_open}")
        print(f"✓ 闭眼预测: {results_closed}")
        
        return True
        
    except Exception as e:
        print(f"❌ 眼部分类器测试失败: {e}")
        return False

def test_audio():
    """测试音频系统"""
    print("测试音频系统...")
    try:
        import pygame
        pygame.mixer.init()
        print("✓ pygame音频系统初始化成功")
        
        # 测试简单音频生成
        import numpy as np
        sample_rate = 22050
        duration = 0.1
        frequency = 440
        
        frames = int(duration * sample_rate)
        arr = np.zeros((frames, 2))
        
        for i in range(frames):
            wave = 32767 * np.sin(frequency * 2 * np.pi * i / sample_rate)
            arr[i][0] = wave
            arr[i][1] = wave
        
        sound = pygame.sndarray.make_sound(arr.astype(np.int16))
        print("✓ 测试音频生成成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 音频系统测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=== 系统组件测试 ===\n")
    
    tests = [
        test_opencv,
        test_pytorch,
        test_dlib,
        test_camera,
        test_face_detection,
        test_eye_classifier,
        test_audio
    ]
    
    results = []
    
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"测试异常: {e}")
            results.append(False)
        print()
    
    # 总结
    passed = sum(results)
    total = len(results)
    
    print("=== 测试结果总结 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！系统已就绪")
        print("\n可以运行以下命令启动系统:")
        print("  python run_demo.py")
        print("  python drowsiness_detector.py")
    else:
        print("⚠️  部分测试失败，请检查相关组件")
        print("\n建议:")
        print("1. 确保所有依赖都已正确安装")
        print("2. 运行 python setup.py 重新设置")
        print("3. 检查摄像头和音频设备连接")

if __name__ == "__main__":
    main()