#!/bin/bash

# Dừng script ngay lập tức nếu có lỗi
set -e

echo ">>> BẮT ĐẦU SETUP..."

# 1. KÍCH HOẠT CONDA TRONG SCRIPT (QUAN TRỌNG NHẤT)
# Dựa vào log của bạn, đường dẫn cài đặt là /opt/miniforge3
if [ -f "/opt/miniforge3/etc/profile.d/conda.sh" ]; then
    source "/opt/miniforge3/etc/profile.d/conda.sh"
else
    echo "Lỗi: Không tìm thấy file conda.sh để kích hoạt!"
    exit 1
fi

# 2. Tạo môi trường Python 3.8 (Nếu chưa có)
# Thêm 'numpy' vào đây luôn để Conda cài bản chuẩn, tránh lỗi build bằng pip
echo ">>> Đang tạo môi trường lsnet..."
conda create -n lsnet python==3.8

# 3. Kích hoạt môi trường
echo ">>> Đang kích hoạt lsnet..."
conda activate lsnet

# Kiểm tra xem đã switch đúng python chưa
echo ">>> Python hiện tại: $(which python)"
echo ">>> Version: $(python --version)"

# 4. Cài các thư viện còn lại
# Lưu ý: Numpy đã cài ở trên rồi, pip sẽ tự bỏ qua hoặc check lại thôi
echo ">>> Đang cài requirements..."
pip install -r requirements.txt

# 5. Tải data
echo ">>> Đang tải dataset..."
curl -L -o dataset/vnfood-30-100.zip \
    https://www.kaggle.com/api/v1/datasets/download/meowluvmatcha/vnfood-30-100

# 6. Giải nén
echo ">>> Đang giải nén..."
mkdir -p dataset
unzip -o -q dataset/vnfood-30-100.zip -d dataset/vnfood-30-100

echo ">>> HOÀN TẤT!"