#!/usr/bin/env python3

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 7):
        print("错误: 需要Python 3.7或更高版本")
        sys.exit(1)
    print(f"✓ Python版本检查通过: {sys.version}")

def install_requirements():
    """安装依赖包"""
    print("正在安装Python依赖包...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ 依赖包安装成功")
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖包安装失败: {e}")
        return False
    return True

def download_dlib_predictor():
    """下载dlib人脸关键点检测器"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    predictor_path = models_dir / "shape_predictor_68_face_landmarks.dat"
    
    if predictor_path.exists():
        print("✓ dlib预测器文件已存在")
        return True
    
    print("正在下载dlib人脸关键点检测器...")
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    
    try:
        import bz2
        import shutil
        
        # 下载压缩文件
        compressed_path = predictor_path.with_suffix('.dat.bz2')
        print(f"下载中: {url}")
        urllib.request.urlretrieve(url, compressed_path)
        
        # 解压缩
        print("解压缩中...")
        with bz2.open(compressed_path, 'rb') as f_in:
            with open(predictor_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # 删除压缩文件
        compressed_path.unlink()
        
        print("✓ dlib预测器下载成功")
        return True
        
    except Exception as e:
        print(f"❌ dlib预测器下载失败: {e}")
        print("请手动下载 shape_predictor_68_face_landmarks.dat 到 models/ 目录")
        return False

def test_camera():
    """测试摄像头访问"""
    print("测试摄像头访问...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("⚠️  无法访问摄像头(ID: 0)")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            print("✓ 摄像头测试成功")
            return True
        else:
            print("⚠️  摄像头读取帧失败")
            return False
            
    except ImportError:
        print("❌ OpenCV未安装")
        return False
    except Exception as e:
        print(f"❌ 摄像头测试失败: {e}")
        return False

def create_directories():
    """创建必要的目录"""
    dirs = ["models", "data", "logs", "output"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    print("✓ 目录结构创建完成")

def main():
    """主安装函数"""
    print("=== 驾驶员疲劳检测系统安装程序 ===\n")
    
    # 检查Python版本
    check_python_version()
    
    # 创建目录
    create_directories()
    
    # 安装依赖
    if not install_requirements():
        print("安装失败，请检查错误信息")
        sys.exit(1)
    
    # 下载dlib预测器
    download_dlib_predictor()
    
    # 测试摄像头
    camera_ok = test_camera()
    
    print("\n=== 安装完成 ===")
    
    if camera_ok:
        print("✅ 系统安装成功！可以运行以下命令启动:")
        print("    python drowsiness_detector.py")
    else:
        print("⚠️  系统安装基本完成，但摄像头测试失败")
        print("   请确保:")
        print("   1. 摄像头已连接并工作正常")
        print("   2. 摄像头驱动已安装")
        print("   3. 没有其他程序占用摄像头")
    
    print("\n其他运行选项:")
    print("    python run_demo.py  # 运行演示")
    print("    python test_system.py  # 测试系统各模块")

if __name__ == "__main__":
    main()