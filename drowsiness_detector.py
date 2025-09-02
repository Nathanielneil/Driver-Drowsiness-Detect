import cv2
import numpy as np
import time
from collections import deque
from face_detection import FaceDetector
from eye_classifier import EyePredictor, create_simple_predictor
from config import DROWSINESS_CONFIG

class DrowsinessDetector:
    def __init__(self, use_cnn=False, model_path=None):
        self.face_detector = FaceDetector()
        
        # 选择预测器类型
        if use_cnn and model_path:
            try:
                self.eye_predictor = EyePredictor(model_path)
                print("使用CNN模型进行眼部状态检测")
            except:
                print("CNN模型加载失败，使用简单规则预测器")
                self.eye_predictor = create_simple_predictor()
        else:
            self.eye_predictor = create_simple_predictor()
            print("使用基于规则的眼部状态检测")
        
        # 疲劳检测参数
        self.ear_threshold = DROWSINESS_CONFIG['ear_threshold']
        self.consecutive_frames = DROWSINESS_CONFIG['consecutive_frames']
        
        # 状态追踪
        self.closed_eye_frames = 0
        self.drowsiness_detected = False
        
        # 历史数据存储
        self.ear_history = deque(maxlen=30)  # 保存最近30帧的EAR值
        self.prediction_history = deque(maxlen=10)  # 保存最近10帧的预测结果
        
        # 统计信息
        self.frame_count = 0
        self.start_time = time.time()
    
    def log_alert(self):
        """记录疲劳警报"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] 警报: 检测到驾驶员疲劳状态！")
    
    def analyze_frame(self, frame):
        self.frame_count += 1
        results = {
            'faces_detected': 0,
            'drowsiness_detected': False,
            'ear_value': None,
            'eye_state': None,
            'confidence': 0.0,
            'alert_triggered': False
        }
        
        # 检测人脸
        faces, gray = self.face_detector.detect_faces(frame)
        results['faces_detected'] = len(faces)
        
        if len(faces) == 0:
            return results, frame
        
        # 处理第一个检测到的人脸
        face = faces[0]
        landmarks = self.face_detector.get_landmarks(gray, face)
        
        if landmarks is not None:
            # 计算EAR值
            left_ear, right_ear = self.face_detector.get_eyes_ear(landmarks)
            if left_ear and right_ear:
                avg_ear = (left_ear + right_ear) / 2.0
                results['ear_value'] = avg_ear
                self.ear_history.append(avg_ear)
                
                # 提取眼部区域进行CNN预测
                left_eye_region, right_eye_region = self.face_detector.extract_eye_regions(frame, landmarks)
                
                # 眼部状态预测
                eye_predictions = self.eye_predictor.predict_both_eyes(left_eye_region, right_eye_region)
                
                if 'combined' in eye_predictions:
                    prediction = eye_predictions['combined']['prediction']
                    confidence = eye_predictions['combined']['confidence']
                    
                    results['eye_state'] = 'closed' if prediction == 0 else 'open'
                    results['confidence'] = confidence
                    
                    self.prediction_history.append(prediction)
                    
                    # 疲劳检测逻辑
                    is_drowsy = self.detect_drowsiness(avg_ear, prediction)
                    results['drowsiness_detected'] = is_drowsy
                    
                    if is_drowsy:
                        results['alert_triggered'] = True
                        self.log_alert()
                
                # 在图像上绘制关键点和信息
                frame = self.draw_annotations(frame, landmarks, results)
        
        return results, frame
    
    def detect_drowsiness(self, ear_value, eye_prediction):
        # 方法1: EAR阈值检测
        ear_drowsy = ear_value < self.ear_threshold
        
        # 方法2: CNN预测检测 (0=闭眼, 1=睁眼)
        cnn_drowsy = eye_prediction == 0
        
        # 综合判断
        is_currently_drowsy = ear_drowsy or cnn_drowsy
        
        if is_currently_drowsy:
            self.closed_eye_frames += 1
        else:
            self.closed_eye_frames = 0
        
        # 连续帧数超过阈值才判断为疲劳
        drowsiness_detected = self.closed_eye_frames >= self.consecutive_frames
        
        if drowsiness_detected != self.drowsiness_detected:
            self.drowsiness_detected = drowsiness_detected
            if drowsiness_detected:
                print("警告: 检测到驾驶员疲劳！")
            else:
                print("正常: 驾驶员状态恢复正常")
        
        return drowsiness_detected
    
    def draw_annotations(self, frame, landmarks, results):
        # 绘制人脸关键点
        frame = self.face_detector.draw_landmarks(frame, landmarks)
        
        # 绘制状态信息
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # EAR值
        if results['ear_value']:
            ear_text = f"EAR: {results['ear_value']:.3f}"
            color = (0, 255, 0) if results['ear_value'] > self.ear_threshold else (0, 0, 255)
            cv2.putText(frame, ear_text, (10, y_offset), font, 0.7, color, 2)
            y_offset += 30
        
        # 眼部状态
        if results['eye_state']:
            state_text = f"Eyes: {results['eye_state']} ({results['confidence']:.2f})"
            color = (0, 255, 0) if results['eye_state'] == 'open' else (0, 0, 255)
            cv2.putText(frame, state_text, (10, y_offset), font, 0.7, color, 2)
            y_offset += 30
        
        # 疲劳状态
        status_text = "DROWSY!" if results['drowsiness_detected'] else "ALERT"
        color = (0, 0, 255) if results['drowsiness_detected'] else (0, 255, 0)
        cv2.putText(frame, status_text, (10, y_offset), font, 1, color, 3)
        y_offset += 40
        
        # 连续闭眼帧数
        frames_text = f"Closed frames: {self.closed_eye_frames}/{self.consecutive_frames}"
        cv2.putText(frame, frames_text, (10, y_offset), font, 0.6, (255, 255, 255), 1)
        y_offset += 25
        
        # FPS
        current_time = time.time()
        fps = self.frame_count / (current_time - self.start_time + 1e-6)
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, y_offset), font, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def get_statistics(self):
        current_time = time.time()
        runtime = current_time - self.start_time
        fps = self.frame_count / (runtime + 1e-6)
        
        stats = {
            'runtime': runtime,
            'total_frames': self.frame_count,
            'average_fps': fps,
            'current_closed_frames': self.closed_eye_frames,
            'drowsiness_detected': self.drowsiness_detected
        }
        
        if self.ear_history:
            stats['average_ear'] = np.mean(self.ear_history)
            stats['current_ear'] = self.ear_history[-1]
        
        return stats

class VideoProcessor:
    def __init__(self, detector, source=0, output_path=None):
        self.detector = detector
        self.source = source
        self.output_path = output_path
        self.cap = None
        self.out = None
    
    def setup_video(self):
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频源: {self.source}")
        
        # 设置视频属性
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 如果需要保存视频
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
    
    def process_video(self):
        self.setup_video()
        
        print("开始疲劳检测...")
        print("按 'q' 退出, 按 's' 显示统计信息")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("视频读取结束或出错")
                    break
                
                # 处理帧
                results, annotated_frame = self.detector.analyze_frame(frame)
                
                # 显示结果
                cv2.imshow('Drowsiness Detection', annotated_frame)
                
                # 保存视频
                if self.out:
                    self.out.write(annotated_frame)
                
                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    stats = self.detector.get_statistics()
                    print("\n=== 统计信息 ===")
                    for key, value in stats.items():
                        print(f"{key}: {value}")
                    print("===============\n")
        
        except KeyboardInterrupt:
            print("\n检测被用户中断")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()
        
        # 显示最终统计
        final_stats = self.detector.get_statistics()
        print("\n=== 最终统计 ===")
        for key, value in final_stats.items():
            print(f"{key}: {value}")
        print("================")

if __name__ == "__main__":
    # 创建检测器
    detector = DrowsinessDetector(use_cnn=False)  # 默认使用简单规则
    
    # 创建视频处理器
    processor = VideoProcessor(detector, source=0)  # 使用摄像头
    
    # 开始处理
    processor.process_video()