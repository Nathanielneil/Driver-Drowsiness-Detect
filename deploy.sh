#!/bin/bash

# 驾驶员疲劳检测系统服务器部署脚本

set -e  # 出错时退出

echo "=== 驾驶员疲劳检测系统服务器部署 ==="

# 检查是否为root用户
if [ "$EUID" -eq 0 ]; then
    echo "警告: 正在以root用户运行"
fi

# 检查系统类型
if [ -f /etc/redhat-release ]; then
    OS="centos"
    echo "检测到CentOS/RHEL系统"
elif [ -f /etc/lsb-release ]; then
    OS="ubuntu"
    echo "检测到Ubuntu系统"
else
    OS="unknown"
    echo "未知系统类型，继续尝试部署..."
fi

# 安装系统依赖
echo "安装系统依赖..."
if [ "$OS" = "ubuntu" ]; then
    sudo apt-get update
    sudo apt-get install -y \
        python3 python3-pip python3-venv \
        build-essential cmake \
        libopencv-dev python3-opencv \
        libdlib-dev \
        ffmpeg \
        wget curl \
        supervisor nginx
elif [ "$OS" = "centos" ]; then
    sudo yum update -y
    sudo yum install -y \
        python3 python3-pip \
        gcc gcc-c++ cmake \
        opencv opencv-devel \
        ffmpeg \
        wget curl \
        supervisor nginx
else
    echo "请手动安装以下依赖: python3, cmake, opencv, ffmpeg"
fi

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python版本: $python_version"

# 创建应用目录和用户
APP_USER="drowsiness"
APP_DIR="/opt/drowsiness-detection"

echo "创建应用用户和目录..."
sudo useradd -r -s /bin/false $APP_USER 2>/dev/null || true
sudo mkdir -p $APP_DIR
sudo chown $APP_USER:$APP_USER $APP_DIR

# 复制项目文件
echo "复制项目文件..."
sudo cp -r . $APP_DIR/
sudo chown -R $APP_USER:$APP_USER $APP_DIR

# 切换到应用目录
cd $APP_DIR

# 创建虚拟环境
echo "创建Python虚拟环境..."
sudo -u $APP_USER python3 -m venv venv
sudo -u $APP_USER $APP_DIR/venv/bin/pip install --upgrade pip

# 安装Python依赖
echo "安装Python依赖..."
sudo -u $APP_USER $APP_DIR/venv/bin/pip install -r requirements.txt

# 下载模型文件
echo "下载dlib模型文件..."
sudo -u $APP_USER $APP_DIR/venv/bin/python setup.py

# 创建日志目录
sudo mkdir -p /var/log/drowsiness-detection
sudo chown $APP_USER:$APP_USER /var/log/drowsiness-detection

# 创建systemd服务文件
echo "创建systemd服务..."
sudo tee /etc/systemd/system/drowsiness-detection.service > /dev/null << 'EOF'
[Unit]
Description=Driver Drowsiness Detection Service
After=network.target

[Service]
Type=simple
User=drowsiness
Group=drowsiness
WorkingDirectory=/opt/drowsiness-detection
Environment=PATH=/opt/drowsiness-detection/venv/bin
ExecStart=/opt/drowsiness-detection/venv/bin/python run_demo.py --source 0
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=drowsiness-detection

# 安全设置
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/drowsiness-detection/logs /opt/drowsiness-detection/output
CapabilityBoundingSet=CAP_NET_BIND_SERVICE

[Install]
WantedBy=multi-user.target
EOF

# 重载systemd并启用服务
sudo systemctl daemon-reload
sudo systemctl enable drowsiness-detection

# 配置防火墙 (如果需要Web访问)
if command -v ufw >/dev/null 2>&1; then
    echo "配置防火墙..."
    sudo ufw allow ssh
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
fi

# 运行测试
echo "运行系统测试..."
sudo -u $APP_USER $APP_DIR/venv/bin/python test_system.py

echo ""
echo "🎉 服务器部署完成！"
echo ""
echo "服务管理命令:"
echo "  sudo systemctl start drowsiness-detection    # 启动服务"
echo "  sudo systemctl stop drowsiness-detection     # 停止服务"
echo "  sudo systemctl status drowsiness-detection   # 查看状态"
echo "  sudo systemctl restart drowsiness-detection  # 重启服务"
echo ""
echo "日志查看:"
echo "  sudo journalctl -u drowsiness-detection -f   # 实时日志"
echo "  sudo journalctl -u drowsiness-detection      # 历史日志"
echo ""
echo "配置文件位置:"
echo "  应用目录: $APP_DIR"
echo "  配置文件: $APP_DIR/config.py"
echo "  日志目录: /var/log/drowsiness-detection"
echo ""
echo "手动运行 (调试用):"
echo "  cd $APP_DIR"
echo "  sudo -u $APP_USER ./venv/bin/python run_demo.py"