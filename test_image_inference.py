#!/usr/bin/env python3
"""
单张图片推理测试脚本
用于测试放入demo文件夹中的图片
"""

import cv2
import json
import argparse
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from face_detection import FaceDetector
from eye_classifier import create_simple_predictor

def test_single_image(image_path, output_dir=None, save_result=True):
    """测试单张图片"""
    
    print(f"测试图片: {image_path}")
    
    # 检查图片是否存在
    if not Path(image_path).exists():
        print(f"错误: 图片文件不存在 - {image_path}")
        return None
    
    # 读取图片
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"错误: 无法读取图片 - {image_path}")
        return None
    
    print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")
    
    # 初始化检测器
    face_detector = FaceDetector()
    eye_predictor = create_simple_predictor()
    
    # 检测人脸
    faces, gray = face_detector.detect_faces(img)
    
    results = {
        'image_path': str(image_path),
        'image_size': f"{img.shape[1]}x{img.shape[0]}",
        'timestamp': datetime.now().isoformat(),
        'faces_detected': len(faces),
        'face_results': []
    }
    
    print(f"检测到 {len(faces)} 个人脸")
    
    if len(faces) == 0:
        print("未检测到人脸")
        results['status'] = 'no_face'
    else:
        results['status'] = 'success'
        
        # 处理每个检测到的人脸
        for i, face in enumerate(faces):
            print(f"\n处理人脸 #{i+1}:")
            
            landmarks = face_detector.get_landmarks(gray, face)
            
            face_result = {
                'face_id': i + 1,
                'landmarks_detected': landmarks is not None
            }
            
            if landmarks is not None:
                print("  ✓ 检测到人脸关键点")
                
                # 计算EAR值
                left_ear, right_ear = face_detector.get_eyes_ear(landmarks)
                if left_ear and right_ear:
                    avg_ear = (left_ear + right_ear) / 2.0
                    face_result['ear_value'] = float(avg_ear)
                    print(f"  EAR值: {avg_ear:.3f}")
                    
                    # 判断眼部状态
                    ear_threshold = 0.25
                    eye_state_by_ear = 'open' if avg_ear > ear_threshold else 'closed'
                    face_result['eye_state_by_ear'] = eye_state_by_ear
                    print(f"  基于EAR的眼部状态: {eye_state_by_ear}")
                    
                    # 提取眼部区域
                    left_eye_region, right_eye_region = face_detector.extract_eye_regions(img, landmarks)
                    
                    if left_eye_region is not None and right_eye_region is not None:
                        # 使用简单预测器
                        eye_predictions = eye_predictor.predict_both_eyes(left_eye_region, right_eye_region)
                        
                        if 'combined' in eye_predictions:
                            prediction = eye_predictions['combined']['prediction']
                            confidence = eye_predictions['combined']['confidence']
                            
                            eye_state_by_cnn = 'open' if prediction == 1 else 'closed'
                            face_result['eye_state_by_cnn'] = eye_state_by_cnn
                            face_result['confidence'] = float(confidence)
                            
                            # print(f"  基于CNN的眼部状态: {eye_state_by_cnn} (置信度: {confidence:.2f})")
                            print(f"  基于CNN的眼部状态: {eye_state_by_cnn} ")
                            
                            # 综合判断疲劳状态
                            is_drowsy = (avg_ear < ear_threshold) or (prediction == 0)
                            face_result['drowsiness_detected'] = bool(is_drowsy)  # 确保转换为Python原生bool类型
                            print(f"  疲劳检测结果: {'疲劳' if is_drowsy else '正常'}")
                    
                # 在图像上绘制检测结果
                img = face_detector.draw_landmarks(img, landmarks)
                
                # 绘制人脸边框
                x, y, w, h = cv2.boundingRect(landmarks)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 添加文字标注
                if 'ear_value' in face_result:
                    ear_text = f"EAR: {face_result['ear_value']:.3f}"
                    cv2.putText(img, ear_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                if 'drowsiness_detected' in face_result:
                    status_text = "DROWSY" if face_result['drowsiness_detected'] else "ALERT"
                    color = (0, 0, 255) if face_result['drowsiness_detected'] else (0, 255, 0)
                    cv2.putText(img, status_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                print("  ❌ 未能检测到人脸关键点")
                face_result['error'] = 'no_landmarks'
            
            results['face_results'].append(face_result)
    
    # 保存结果
    if save_result:
        if output_dir is None:
            output_dir = Path("output")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # 保存处理后的图像
        image_name = Path(image_path).stem
        result_image_path = output_dir / f"result_{image_name}.jpg"
        cv2.imwrite(str(result_image_path), img)
        results['result_image'] = str(result_image_path)
        print(f"\n处理后的图像已保存: {result_image_path}")
        
        # 保存JSON结果
        json_path = output_dir / f"result_{image_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            # 使用自定义序列化函数确保numpy类型正确转换
            json.dump(results, f, indent=2, ensure_ascii=False, default=lambda obj: bool(obj) if isinstance(obj, np.bool_) else obj)
        results['result_json'] = str(json_path)
        print(f"检测结果已保存: {json_path}")
    
    return results

def batch_test_demo_folder(demo_folder="demo", output_dir="output"):
    """批量测试demo文件夹中的所有图片"""
    
    demo_path = Path(demo_folder)
    if not demo_path.exists():
        print(f"错误: demo文件夹不存在 - {demo_folder}")
        print("请创建demo文件夹并放入测试图片")
        return []
    
    # 查找所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(demo_path.glob(f"*{ext}"))
        image_files.extend(demo_path.glob(f"*{ext.upper()}"))
        # 递归搜索子文件夹
        image_files.extend(demo_path.glob(f"**/*{ext}"))
        image_files.extend(demo_path.glob(f"**/*{ext.upper()}"))
    
    if not image_files:
        print(f"在 {demo_folder} 文件夹中没有找到图片文件")
        print(f"支持的格式: {', '.join(image_extensions)}")
        return []
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    all_results = []
    successful_tests = 0
    
    for img_path in sorted(set(image_files)):  # 使用set去重
        print(f"\n{'='*50}")
        try:
            result = test_single_image(img_path, output_dir)
            if result:
                all_results.append(result)
                if result['status'] == 'success':
                    successful_tests += 1
        except Exception as e:
            print(f"处理 {img_path} 时出错: {e}")
    
    # 生成总结报告
    summary = {
        'test_time': datetime.now().isoformat(),
        'total_images': len(image_files),
        'successful_tests': successful_tests,
        'failed_tests': len(image_files) - successful_tests,
        'face_detection_rate': f"{sum(1 for r in all_results if r['faces_detected'] > 0) / len(all_results) * 100:.1f}%" if all_results else "0%",
        'drowsiness_detection_count': sum(1 for r in all_results for f in r.get('face_results', []) if f.get('drowsiness_detected', False)),
        'detailed_results': all_results
    }
    
    # 保存总结报告
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    summary_path = output_path / "batch_test_summary.json"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        # 使用自定义序列化函数确保numpy类型正确转换
        json.dump(summary, f, indent=2, ensure_ascii=False, default=lambda obj: bool(obj) if isinstance(obj, np.bool_) else obj)
    
    print(f"\n{'='*50}")
    print("批量测试完成！")
    print(f"总图片数: {summary['total_images']}")
    print(f"成功处理: {summary['successful_tests']}")
    print(f"处理失败: {summary['failed_tests']}")
    print(f"人脸检测率: {summary['face_detection_rate']}")
    print(f"检测到疲劳的人脸数: {summary['drowsiness_detection_count']}")
    print(f"详细报告: {summary_path}")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='图片推理测试工具')
    parser.add_argument('--image', type=str, help='单张图片路径')
    parser.add_argument('--demo-folder', type=str, default='demo', help='demo文件夹路径')
    parser.add_argument('--output', type=str, default='output', help='输出文件夹路径')
    parser.add_argument('--batch', action='store_true', help='批量测试demo文件夹中的所有图片')
    
    args = parser.parse_args()
    
    if args.image:
        # 测试单张图片
        result = test_single_image(args.image, args.output)
        if result:
            print(f"\n测试完成: {result['status']}")
        else:
            sys.exit(1)
    elif args.batch:
        # 批量测试
        batch_test_demo_folder(args.demo_folder, args.output)
    else:
        # 默认批量测试demo文件夹
        batch_test_demo_folder(args.demo_folder, args.output)

if __name__ == "__main__":
    main()