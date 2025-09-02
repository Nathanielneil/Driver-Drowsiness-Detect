import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from config import DLIB_CONFIG, EYE_LANDMARKS

class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = None
        self.load_predictor()
    
    def load_predictor(self):
        try:
            self.predictor = dlib.shape_predictor(DLIB_CONFIG['predictor_path'])
            print("dlib人脸关键点检测器加载成功")
        except Exception as e:
            print(f"警告: dlib预测器加载失败: {e}")
            print("请下载 shape_predictor_68_face_landmarks.dat 到 models/ 目录")
    
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        return faces, gray
    
    def get_landmarks(self, gray, face):
        if self.predictor is None:
            return None
        landmarks = self.predictor(gray, face)
        landmarks_points = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
        return np.array(landmarks_points)
    
    def extract_eye_regions(self, frame, landmarks):
        if landmarks is None:
            return None, None
        
        left_eye_points = landmarks[EYE_LANDMARKS['left_eye']]
        right_eye_points = landmarks[EYE_LANDMARKS['right_eye']]
        
        # 计算眼部区域边界框
        left_eye_region = self._extract_eye_region(frame, left_eye_points)
        right_eye_region = self._extract_eye_region(frame, right_eye_points)
        
        return left_eye_region, right_eye_region
    
    def _extract_eye_region(self, frame, eye_points):
        x, y, w, h = cv2.boundingRect(eye_points)
        # 扩展边界框
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(frame.shape[1] - x, w + 2 * margin)
        h = min(frame.shape[0] - y, h + 2 * margin)
        
        eye_region = frame[y:y+h, x:x+w]
        return eye_region
    
    def calculate_ear(self, eye_points):
        # 计算眼部纵横比 (Eye Aspect Ratio)
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        C = dist.euclidean(eye_points[0], eye_points[3])
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def get_eyes_ear(self, landmarks):
        if landmarks is None:
            return None, None
        
        left_eye = landmarks[EYE_LANDMARKS['left_eye']]
        right_eye = landmarks[EYE_LANDMARKS['right_eye']]
        
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        
        return left_ear, right_ear
    
    def draw_landmarks(self, frame, landmarks):
        if landmarks is None:
            return frame
        
        for point in landmarks:
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
        
        # 高亮眼部区域
        left_eye = landmarks[EYE_LANDMARKS['left_eye']]
        right_eye = landmarks[EYE_LANDMARKS['right_eye']]
        
        cv2.polylines(frame, [left_eye], True, (0, 255, 255), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 255), 1)
        
        return frame