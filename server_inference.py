#!/usr/bin/env python3
"""
服务器推理服务
适用于无显示器的服务器环境
"""

import cv2
import numpy as np
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from face_detection import FaceDetector
from eye_classifier import create_simple_predictor
from config import DROWSINESS_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/inference.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ServerInferenceEngine:
    def __init__(self, model_path=None, save_results=True):
        self.face_detector = FaceDetector()
        self.eye_predictor = create_simple_predictor()
        self.save_results = save_results
        
        # 疲劳检测参数
        self.ear_threshold = DROWSINESS_CONFIG['ear_threshold']
        self.consecutive_frames = DROWSINESS_CONFIG['consecutive_frames']
        
        # 状态追踪
        self.closed_eye_frames = 0
        self.drowsiness_detected = False
        self.frame_count = 0
        self.start_time = time.time()
        
        # 结果存储
        self.results_buffer = []
        self.alert_log = []
        
        logger.info("服务器推理引擎初始化完成")
    
    def process_frame(self, frame):
        """处理单帧图像"""
        self.frame_count += 1
        timestamp = time.time()
        
        results = {
            'timestamp': timestamp,
            'frame_id': self.frame_count,
            'faces_detected': 0,
            'drowsiness_detected': False,
            'ear_value': None,
            'eye_state': None,
            'confidence': 0.0,
            'processing_time_ms': 0
        }
        
        start_processing = time.time()
        
        try:
            # 检测人脸
            faces, gray = self.face_detector.detect_faces(frame)
            results['faces_detected'] = len(faces)
            
            if len(faces) > 0:
                # 处理第一个检测到的人脸
                face = faces[0]
                landmarks = self.face_detector.get_landmarks(gray, face)
                
                if landmarks is not None:
                    # 计算EAR值
                    left_ear, right_ear = self.face_detector.get_eyes_ear(landmarks)
                    if left_ear and right_ear:
                        avg_ear = (left_ear + right_ear) / 2.0
                        results['ear_value'] = float(avg_ear)
                        
                        # 提取眼部区域
                        left_eye_region, right_eye_region = self.face_detector.extract_eye_regions(frame, landmarks)
                        
                        # 眼部状态预测
                        eye_predictions = self.eye_predictor.predict_both_eyes(left_eye_region, right_eye_region)
                        
                        if 'combined' in eye_predictions:
                            prediction = eye_predictions['combined']['prediction']
                            confidence = eye_predictions['combined']['confidence']
                            
                            results['eye_state'] = 'closed' if prediction == 0 else 'open'
                            results['confidence'] = float(confidence)
                            
                            # 疲劳检测
                            is_drowsy = self.detect_drowsiness(avg_ear, prediction)
                            results['drowsiness_detected'] = is_drowsy
                            
                            if is_drowsy:
                                self.log_alert(results)
        
        except Exception as e:
            logger.error(f"帧处理错误: {e}")
            results['error'] = str(e)
        
        # 计算处理时间
        results['processing_time_ms'] = (time.time() - start_processing) * 1000
        
        # 保存结果
        if self.save_results:
            self.results_buffer.append(results)
        
        return results
    
    def detect_drowsiness(self, ear_value, eye_prediction):
        """疲劳检测逻辑"""
        ear_drowsy = ear_value < self.ear_threshold
        cnn_drowsy = eye_prediction == 0
        
        is_currently_drowsy = ear_drowsy or cnn_drowsy
        
        if is_currently_drowsy:
            self.closed_eye_frames += 1
        else:
            self.closed_eye_frames = 0
        
        drowsiness_detected = self.closed_eye_frames >= self.consecutive_frames
        
        if drowsiness_detected != self.drowsiness_detected:
            self.drowsiness_detected = drowsiness_detected
            if drowsiness_detected:
                logger.warning("⚠️  检测到驾驶员疲劳！")
            else:
                logger.info("✅ 驾驶员状态恢复正常")
        
        return drowsiness_detected
    
    def log_alert(self, results):
        """记录警报"""
        alert = {
            'timestamp': results['timestamp'],
            'datetime': datetime.fromtimestamp(results['timestamp']).isoformat(),
            'frame_id': results['frame_id'],
            'ear_value': results['ear_value'],
            'eye_state': results['eye_state'],
            'confidence': results['confidence']
        }
        
        self.alert_log.append(alert)
        logger.warning(f"疲劳警报: {alert}")
    
    def get_statistics(self):
        """获取统计信息"""
        current_time = time.time()
        runtime = current_time - self.start_time
        fps = self.frame_count / (runtime + 1e-6)
        
        stats = {
            'runtime_seconds': runtime,
            'total_frames': self.frame_count,
            'average_fps': fps,
            'current_closed_frames': self.closed_eye_frames,
            'drowsiness_detected': self.drowsiness_detected,
            'total_alerts': len(self.alert_log)
        }
        
        return stats
    
    def save_results_to_file(self, filepath=None):
        """保存结果到文件"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"output/inference_results_{timestamp}.json"
        
        data = {
            'statistics': self.get_statistics(),
            'alerts': self.alert_log,
            'config': {
                'ear_threshold': self.ear_threshold,
                'consecutive_frames': self.consecutive_frames
            }
        }
        
        # 如果需要详细结果，也可以包含
        if len(self.results_buffer) < 10000:  # 避免文件过大
            data['detailed_results'] = self.results_buffer
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已保存到: {filepath}")
        return filepath

class VideoStreamProcessor:
    def __init__(self, inference_engine, source=0, batch_size=1):
        self.engine = inference_engine
        self.source = source
        self.batch_size = batch_size
        self.cap = None
        
    def setup_capture(self):
        """设置视频捕获"""
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频源: {self.source}")
        
        # 设置视频参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info(f"视频捕获设置完成: {self.source}")
    
    def process_stream(self, duration_seconds=None, save_interval=300):
        """处理视频流"""
        self.setup_capture()
        
        start_time = time.time()
        last_save_time = start_time
        
        logger.info("开始处理视频流...")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("无法读取视频帧")
                    break
                
                # 处理帧
                results = self.engine.process_frame(frame)
                
                # 定期保存结果
                current_time = time.time()
                if current_time - last_save_time > save_interval:
                    self.engine.save_results_to_file()
                    last_save_time = current_time
                    
                    # 输出统计信息
                    stats = self.engine.get_statistics()
                    logger.info(f"统计信息: FPS={stats['average_fps']:.1f}, "
                              f"总帧数={stats['total_frames']}, "
                              f"警报数={stats['total_alerts']}")
                
                # 检查运行时间
                if duration_seconds and (current_time - start_time) > duration_seconds:
                    logger.info(f"达到设定运行时间: {duration_seconds}秒")
                    break
                
                # 短暂休眠以控制CPU使用率
                time.sleep(0.001)
        
        except KeyboardInterrupt:
            logger.info("检测被用户中断")
        except Exception as e:
            logger.error(f"流处理错误: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        if self.cap:
            self.cap.release()
        
        # 保存最终结果
        filepath = self.engine.save_results_to_file()
        
        # 输出最终统计
        stats = self.engine.get_statistics()
        logger.info("=== 最终统计 ===")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")

def main():
    parser = argparse.ArgumentParser(description='服务器端疲劳检测推理服务')
    
    parser.add_argument('--source', type=str, default='0',
                       help='视频源 (默认: 0)')
    parser.add_argument('--duration', type=int, default=None,
                       help='运行时长(秒) (默认: 无限制)')
    parser.add_argument('--save-interval', type=int, default=300,
                       help='结果保存间隔(秒) (默认: 300)')
    parser.add_argument('--ear-threshold', type=float, default=0.25,
                       help='EAR阈值 (默认: 0.25)')
    parser.add_argument('--consecutive-frames', type=int, default=20,
                       help='连续帧阈值 (默认: 20)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别 (默认: INFO)')
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 转换源参数
    source = args.source
    if source.isdigit():
        source = int(source)
    
    logger.info("=== 服务器端疲劳检测推理服务 ===")
    logger.info(f"视频源: {source}")
    logger.info(f"运行时长: {args.duration or '无限制'}秒")
    logger.info(f"保存间隔: {args.save_interval}秒")
    logger.info(f"EAR阈值: {args.ear_threshold}")
    logger.info(f"连续帧阈值: {args.consecutive_frames}")
    
    try:
        # 创建推理引擎
        engine = ServerInferenceEngine()
        engine.ear_threshold = args.ear_threshold
        engine.consecutive_frames = args.consecutive_frames
        
        # 创建流处理器
        processor = VideoStreamProcessor(engine, source)
        
        # 开始处理
        processor.process_stream(
            duration_seconds=args.duration,
            save_interval=args.save_interval
        )
        
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())