# Demo测试图片说明

这个文件夹用于存放测试图片，以便在没有摄像头或视频输入时测试疲劳检测系统。

## 建议的测试图片类型

请在此文件夹中放入以下类型的图片进行测试：

### 1. 正常状态图片
- `normal_driver_1.jpg` - 驾驶员睁眼，正常状态
- `normal_driver_2.jpg` - 不同角度的正常驾驶员
- `alert_face.jpg` - 清醒状态的面部图像

### 2. 疲劳状态图片  
- `drowsy_driver_1.jpg` - 驾驶员闭眼或半闭眼
- `drowsy_driver_2.jpg` - 疲劳状态的面部表情
- `sleepy_face.jpg` - 困倦的面部图像

### 3. 边界情况测试
- `partial_closed.jpg` - 部分闭眼状态
- `lowlight_driver.jpg` - 低光照条件下的图像
- `side_angle.jpg` - 侧面角度的人脸
- `multiple_faces.jpg` - 包含多个人脸的图像

### 4. 异常情况测试
- `no_face.jpg` - 不包含人脸的图像
- `blurry_face.jpg` - 模糊的人脸图像
- `occluded_face.jpg` - 部分遮挡的人脸

## 图片要求

- **格式**: 支持 .jpg, .png, .bmp 等常见格式
- **尺寸**: 推荐 640x480 或更高分辨率
- **质量**: 尽量使用清晰的图像，避免过度压缩
- **光照**: 包含不同光照条件的图像以测试鲁棒性

## 使用方法

### 1. 批量测试所有图片
```bash
python3 test_demo_images.py --mode batch
```

### 2. 交互式测试单张图片
```bash
python3 test_demo_images.py --mode interactive
```

### 3. 测试指定图片
```bash
# 作为视频源使用图片
python3 run_demo.py --source demo/normal_driver_1.jpg

# 服务器推理测试
python3 server_inference.py --source demo/drowsy_driver_1.jpg --duration 10
```

### 4. 生成对比网格
```bash
python3 test_demo_images.py --comparison-grid
```

## 测试结果

测试完成后，结果会保存在以下位置：

- `output/result_*.jpg` - 处理后的图像（标注了检测结果）
- `output/demo_test_report.json` - 详细的测试报告
- `output/comparison_grid.jpg` - 原图与处理结果的对比网格
- `logs/inference.log` - 详细的处理日志

## 测试指标说明

系统会输出以下检测指标：

- **faces_detected**: 检测到的人脸数量
- **ear_value**: 眼部纵横比 (Eye Aspect Ratio)
  - 正常睁眼: > 0.25
  - 闭眼状态: < 0.25
- **eye_state**: 眼部状态 (open/closed)
- **confidence**: 预测置信度 (0-1)
- **drowsiness_detected**: 是否检测到疲劳

## 添加新图片

1. 将图片文件复制到此文件夹
2. 确保文件名有意义，便于识别测试场景
3. 运行测试脚本验证检测效果

## 注意事项

- 图片文件名请使用英文，避免特殊字符
- 建议包含多样化的测试场景以全面评估系统性能
- 可以使用子文件夹组织不同类型的测试图片
- 测试完成后检查 `output/` 目录中的结果文件

## 示例文件结构

```
demo/
├── README.md                 # 本说明文件
├── normal/                   # 正常状态图片
│   ├── alert_driver_1.jpg
│   ├── alert_driver_2.jpg
│   └── normal_lighting.jpg
├── drowsy/                   # 疲劳状态图片
│   ├── closed_eyes_1.jpg
│   ├── sleepy_face.jpg
│   └── drowsy_driver.jpg
├── edge_cases/               # 边界情况
│   ├── partial_closed.jpg
│   ├── side_angle.jpg
│   └── multiple_faces.jpg
└── problematic/              # 异常情况
    ├── no_face.jpg
    ├── blurry.jpg
    └── low_light.jpg
```