#!/usr/bin/env python3
"""
驾驶员疲劳检测演示程序
"""

import argparse
import sys
from drowsiness_detector import DrowsinessDetector, VideoProcessor

def main():
    parser = argparse.ArgumentParser(description='驾驶员疲劳检测演示')
    
    parser.add_argument('--source', type=str, default='0',
                       help='视频源 (0=摄像头, 或视频文件路径)')
    
    parser.add_argument('--output', type=str, default=None,
                       help='输出视频文件路径 (可选)')
    
    parser.add_argument('--model', type=str, default=None,
                       help='CNN模型路径 (可选，默认使用规则预测)')
    
    parser.add_argument('--ear-threshold', type=float, default=0.25,
                       help='EAR阈值 (默认: 0.25)')
    
    parser.add_argument('--consecutive-frames', type=int, default=20,
                       help='连续闭眼帧数阈值 (默认: 20)')
    
    args = parser.parse_args()
    
    # 转换源参数
    source = args.source
    if source.isdigit():
        source = int(source)
    
    print("=== 驾驶员疲劳检测系统 ===")
    print(f"视频源: {source}")
    print(f"输出文件: {args.output or '无'}")
    print(f"使用模型: {args.model or '基于规则的检测'}")
    print(f"EAR阈值: {args.ear_threshold}")
    print(f"连续帧阈值: {args.consecutive_frames}")
    print()
    
    try:
        # 创建检测器
        use_cnn = args.model is not None
        detector = DrowsinessDetector(use_cnn=use_cnn, model_path=args.model)
        
        # 更新配置参数
        detector.ear_threshold = args.ear_threshold
        detector.consecutive_frames = args.consecutive_frames
        
        # 创建视频处理器
        processor = VideoProcessor(detector, source=source, output_path=args.output)
        
        # 开始检测
        processor.process_video()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()