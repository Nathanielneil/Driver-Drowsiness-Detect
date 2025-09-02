#!/usr/bin/env python3
"""
创建测试图片的脚本
用于生成模拟的驾驶员面部图像进行测试
"""

import cv2
import numpy as np
import os
from pathlib import Path

def create_test_images():
    """创建测试用的模拟图像"""
    
    # 创建demo目录
    demo_dir = Path("demo")
    demo_dir.mkdir(exist_ok=True)
    
    print("正在创建测试图像...")
    
    # 图像尺寸
    width, height = 640, 480
    
    # 1. 正常驾驶员图像 (睁眼)
    print("创建正常驾驶员图像...")
    normal_img = create_face_image(width, height, eyes_open=True, brightness=180)
    cv2.imwrite(str(demo_dir / "normal_driver.jpg"), normal_img)
    
    # 2. 疲劳驾驶员图像 (闭眼)
    print("创建疲劳驾驶员图像...")
    drowsy_img = create_face_image(width, height, eyes_open=False, brightness=160)
    cv2.imwrite(str(demo_dir / "drowsy_driver.jpg"), drowsy_img)
    
    # 3. 部分闭眼图像
    print("创建部分闭眼图像...")
    partial_img = create_face_image(width, height, eyes_open=True, brightness=170, eye_height=0.3)
    cv2.imwrite(str(demo_dir / "partial_closed.jpg"), partial_img)
    
    # 4. 低光照图像
    print("创建低光照图像...")
    lowlight_img = create_face_image(width, height, eyes_open=True, brightness=80)
    cv2.imwrite(str(demo_dir / "lowlight_driver.jpg"), lowlight_img)
    
    # 5. 多人图像
    print("创建多人图像...")
    multi_img = create_multi_face_image(width, height)
    cv2.imwrite(str(demo_dir / "multi_person.jpg"), multi_img)
    
    # 6. 无人脸图像
    print("创建无人脸图像...")
    no_face_img = np.random.randint(0, 100, (height, width, 3), dtype=np.uint8)
    cv2.putText(no_face_img, "NO FACE", (width//2-80, height//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.imwrite(str(demo_dir / "no_face.jpg"), no_face_img)
    
    print(f"✓ 测试图像已创建在 {demo_dir} 目录")
    return demo_dir

def create_face_image(width, height, eyes_open=True, brightness=180, eye_height=1.0):
    """创建简单的人脸图像"""
    
    # 创建背景
    img = np.ones((height, width, 3), dtype=np.uint8) * 50
    
    # 添加一些纹理
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # 脸部轮廓 (椭圆)
    face_center = (width // 2, height // 2)
    face_axes = (120, 150)
    cv2.ellipse(img, face_center, face_axes, 0, 0, 360, 
                (brightness-20, brightness-10, brightness), -1)
    
    # 眼部位置
    left_eye_center = (face_center[0] - 40, face_center[1] - 30)
    right_eye_center = (face_center[0] + 40, face_center[1] - 30)
    
    if eyes_open:
        # 睁开的眼睛
        eye_width, eye_height_val = 25, int(15 * eye_height)
        
        # 眼白
        cv2.ellipse(img, left_eye_center, (eye_width, eye_height_val), 0, 0, 360, 
                    (255, 255, 255), -1)
        cv2.ellipse(img, right_eye_center, (eye_width, eye_height_val), 0, 0, 360, 
                    (255, 255, 255), -1)
        
        # 瞳孔
        cv2.circle(img, left_eye_center, 8, (0, 0, 0), -1)
        cv2.circle(img, right_eye_center, 8, (0, 0, 0), -1)
        
        # 眼睛轮廓
        cv2.ellipse(img, left_eye_center, (eye_width, eye_height_val), 0, 0, 360, 
                    (0, 0, 0), 2)
        cv2.ellipse(img, right_eye_center, (eye_width, eye_height_val), 0, 0, 360, 
                    (0, 0, 0), 2)
    else:
        # 闭眼 (画线)
        cv2.line(img, (left_eye_center[0]-25, left_eye_center[1]), 
                 (left_eye_center[0]+25, left_eye_center[1]), (0, 0, 0), 3)
        cv2.line(img, (right_eye_center[0]-25, right_eye_center[1]), 
                 (right_eye_center[0]+25, right_eye_center[1]), (0, 0, 0), 3)
    
    # 鼻子
    nose_points = np.array([
        [face_center[0], face_center[1] + 10],
        [face_center[0] - 8, face_center[1] + 25],
        [face_center[0] + 8, face_center[1] + 25]
    ])
    cv2.fillPoly(img, [nose_points], (brightness-30, brightness-20, brightness-10))
    
    # 嘴巴
    mouth_center = (face_center[0], face_center[1] + 50)
    cv2.ellipse(img, mouth_center, (20, 8), 0, 0, 180, (50, 50, 50), -1)
    
    # 添加一些面部特征点 (模拟68点标记中的关键点)
    # 下巴轮廓
    for i in range(17):
        angle = (i - 8) * 0.2
        x = int(face_center[0] + 100 * np.sin(angle))
        y = int(face_center[1] + 80 + 50 * np.cos(angle))
        cv2.circle(img, (x, y), 1, (brightness+20, brightness+20, brightness+20), -1)
    
    return img

def create_multi_face_image(width, height):
    """创建多人脸图像"""
    img = np.ones((height, width, 3), dtype=np.uint8) * 60
    
    # 添加背景纹理
    noise = np.random.randint(0, 40, (height, width, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # 第一个人脸 (左侧，睁眼)
    face1_center = (width // 3, height // 2)
    face1_axes = (80, 100)
    cv2.ellipse(img, face1_center, face1_axes, 0, 0, 360, (170, 160, 150), -1)
    
    # 第一个人的眼睛 (睁眼)
    left_eye1 = (face1_center[0] - 25, face1_center[1] - 20)
    right_eye1 = (face1_center[0] + 25, face1_center[1] - 20)
    cv2.ellipse(img, left_eye1, (15, 10), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, right_eye1, (15, 10), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, left_eye1, 5, (0, 0, 0), -1)
    cv2.circle(img, right_eye1, 5, (0, 0, 0), -1)
    
    # 第二个人脸 (右侧，闭眼)
    face2_center = (width * 2 // 3, height // 2)
    face2_axes = (80, 100)
    cv2.ellipse(img, face2_center, face2_axes, 0, 0, 360, (160, 150, 140), -1)
    
    # 第二个人的眼睛 (闭眼)
    left_eye2 = (face2_center[0] - 25, face2_center[1] - 20)
    right_eye2 = (face2_center[0] + 25, face2_center[1] - 20)
    cv2.line(img, (left_eye2[0]-15, left_eye2[1]), (left_eye2[0]+15, left_eye2[1]), (0, 0, 0), 2)
    cv2.line(img, (right_eye2[0]-15, right_eye2[1]), (right_eye2[0]+15, right_eye2[1]), (0, 0, 0), 2)
    
    return img

if __name__ == "__main__":
    demo_dir = create_test_images()
    
    print("\n创建的测试图像:")
    for img_file in demo_dir.glob("*.jpg"):
        print(f"  - {img_file.name}")
    
    print(f"\n可以使用以下命令测试:")
    print(f"  python3 test_demo_images.py")
    print(f"  python3 run_demo.py --source demo/normal_driver.jpg")