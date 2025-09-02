#!/usr/bin/env python3
"""
测试demo图片的脚本
批量处理demo文件夹中的测试图像
"""

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from face_detection import FaceDetector
from eye_classifier import create_simple_predictor
from drowsiness_detector import DrowsinessDetector

def test_single_image(image_path, detector):
    """测试单张图像"""
    print(f"\n测试图像: {image_path.name}")
    
    # 读取图像
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ 无法读取图像: {image_path}")
        return None
    
    print(f"  图像尺寸: {img.shape[1]}x{img.shape[0]}")
    
    # 处理图像
    results, annotated_img = detector.analyze_frame(img)
    
    # 显示结果
    print(f"  检测到人脸: {results['faces_detected']}")
    if results['ear_value']:
        print(f"  EAR值: {results['ear_value']:.3f}")
    if results['eye_state']:
        print(f"  眼部状态: {results['eye_state']} (置信度: {results['confidence']:.2f})")
    print(f"  疲劳检测: {'是' if results['drowsiness_detected'] else '否'}")
    
    # 保存处理结果
    output_path = Path("output") / f"result_{image_path.stem}.jpg"
    cv2.imwrite(str(output_path), annotated_img)
    print(f"  结果已保存: {output_path}")
    
    return results

def batch_test_images():
    """批量测试demo图像"""
    print("=== Demo图像批量测试 ===")
    
    # 检查demo目录
    demo_dir = Path("demo")
    if not demo_dir.exists():
        print("demo目录不存在，正在创建测试图像...")
        from create_test_images import create_test_images
        create_test_images()
    
    # 获取所有图像文件
    image_files = list(demo_dir.glob("*.jpg")) + list(demo_dir.glob("*.png"))
    if not image_files:
        print("demo目录中没有找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个测试图像")
    
    # 创建检测器
    detector = DrowsinessDetector(use_cnn=False)
    
    # 存储所有结果
    all_results = []
    
    # 逐个处理图像
    for img_path in sorted(image_files):
        result = test_single_image(img_path, detector)
        if result:
            result['image_name'] = img_path.name
            all_results.append(result)
    
    # 生成测试报告
    generate_test_report(all_results)
    
    print(f"\n✓ 批量测试完成，共处理 {len(all_results)} 张图像")
    print("  详细结果请查看: output/demo_test_report.json")
    print("  处理后的图像在: output/result_*.jpg")

def generate_test_report(results):
    """生成测试报告"""
    report = {
        'test_time': datetime.now().isoformat(),
        'total_images': len(results),
        'summary': {
            'faces_detected_count': sum(1 for r in results if r['faces_detected'] > 0),
            'drowsiness_detected_count': sum(1 for r in results if r['drowsiness_detected']),
            'average_confidence': np.mean([r['confidence'] for r in results if r['confidence'] > 0])
        },
        'detailed_results': results
    }
    
    # 分析结果
    face_detection_rate = report['summary']['faces_detected_count'] / len(results) * 100
    drowsiness_rate = report['summary']['drowsiness_detected_count'] / len(results) * 100
    
    report['analysis'] = {
        'face_detection_rate': f"{face_detection_rate:.1f}%",
        'drowsiness_detection_rate': f"{drowsiness_rate:.1f}%",
        'average_confidence': f"{report['summary']['average_confidence']:.2f}"
    }
    
    # 保存报告
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / "demo_test_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print(f"\n=== 测试摘要 ===")
    print(f"总图像数: {report['total_images']}")
    print(f"人脸检测率: {report['analysis']['face_detection_rate']}")
    print(f"疲劳检测率: {report['analysis']['drowsiness_detection_rate']}")
    print(f"平均置信度: {report['analysis']['average_confidence']}")

def create_comparison_grid():
    """创建对比网格图像"""
    print("\n创建对比网格...")
    
    demo_dir = Path("demo")
    output_dir = Path("output")
    
    # 获取原始图像和结果图像
    original_images = []
    result_images = []
    
    for img_path in sorted(demo_dir.glob("*.jpg")):
        result_path = output_dir / f"result_{img_path.stem}.jpg"
        
        if img_path.exists() and result_path.exists():
            orig_img = cv2.imread(str(img_path))
            result_img = cv2.imread(str(result_path))
            
            # 调整大小
            orig_img = cv2.resize(orig_img, (320, 240))
            result_img = cv2.resize(result_img, (320, 240))
            
            # 添加标题
            cv2.putText(orig_img, f"Original: {img_path.stem}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(result_img, f"Processed: {img_path.stem}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            original_images.append(orig_img)
            result_images.append(result_img)
    
    if not original_images:
        print("没有找到可用的图像对")
        return
    
    # 创建网格
    rows = len(original_images)
    cols = 2  # 原始图像和处理后图像
    
    # 计算网格尺寸
    grid_height = rows * 240
    grid_width = cols * 320
    
    grid_img = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # 填充网格
    for i in range(rows):
        y = i * 240
        
        # 原始图像
        grid_img[y:y+240, 0:320] = original_images[i]
        
        # 处理后图像
        if i < len(result_images):
            grid_img[y:y+240, 320:640] = result_images[i]
    
    # 保存网格图像
    grid_path = output_dir / "comparison_grid.jpg"
    cv2.imwrite(str(grid_path), grid_img)
    print(f"对比网格已保存: {grid_path}")

def interactive_demo():
    """交互式demo"""
    print("\n=== 交互式Demo ===")
    
    demo_dir = Path("demo")
    if not demo_dir.exists():
        print("demo目录不存在，正在创建...")
        from create_test_images import create_test_images
        create_test_images()
    
    detector = DrowsinessDetector(use_cnn=False)
    
    image_files = list(demo_dir.glob("*.jpg"))
    
    print("\n可用的测试图像:")
    for i, img_path in enumerate(image_files):
        print(f"  {i+1}. {img_path.name}")
    
    while True:
        try:
            choice = input(f"\n选择图像 (1-{len(image_files)}) 或输入 'q' 退出: ").strip()
            
            if choice.lower() == 'q':
                break
            
            idx = int(choice) - 1
            if 0 <= idx < len(image_files):
                img_path = image_files[idx]
                
                # 处理图像
                img = cv2.imread(str(img_path))
                results, annotated_img = detector.analyze_frame(img)
                
                # 显示结果窗口 (如果有显示器)
                try:
                    cv2.imshow('Original', img)
                    cv2.imshow('Processed', annotated_img)
                    print("按任意键继续...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except:
                    print("无显示器，跳过图像显示")
                
                # 保存结果
                output_path = Path("output") / f"interactive_result_{img_path.stem}.jpg"
                cv2.imwrite(str(output_path), annotated_img)
                
                # 显示文本结果
                print(f"\n结果:")
                print(f"  人脸检测: {results['faces_detected']}")
                print(f"  EAR值: {results['ear_value']:.3f}" if results['ear_value'] else "  EAR值: 无")
                print(f"  眼部状态: {results['eye_state']}" if results['eye_state'] else "  眼部状态: 未检测")
                print(f"  疲劳状态: {'检测到疲劳' if results['drowsiness_detected'] else '正常'}")
                print(f"  结果已保存: {output_path}")
            else:
                print("无效选择")
                
        except ValueError:
            print("请输入有效数字")
        except KeyboardInterrupt:
            break
    
    print("退出交互式demo")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo图像测试工具')
    parser.add_argument('--mode', choices=['batch', 'interactive', 'create'], 
                       default='batch', help='运行模式')
    parser.add_argument('--create-images', action='store_true',
                       help='创建测试图像')
    parser.add_argument('--comparison-grid', action='store_true',
                       help='创建对比网格')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    Path("output").mkdir(exist_ok=True)
    
    if args.create_images or args.mode == 'create':
        from create_test_images import create_test_images
        create_test_images()
        return
    
    if args.mode == 'batch':
        batch_test_images()
        if args.comparison_grid:
            create_comparison_grid()
    elif args.mode == 'interactive':
        interactive_demo()

if __name__ == "__main__":
    main()