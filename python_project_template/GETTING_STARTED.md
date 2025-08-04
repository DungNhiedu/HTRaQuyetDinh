# 🚀 Hướng Dẫn Sử Dụng Stock Market Prediction System

## 📋 Yêu Cầu Hệ Thống

- **Python**: 3.8 hoặc cao hơn
- **Hệ điều hành**: Windows, macOS, hoặc Linux
- **Ram**: Tối thiểu 4GB (khuyến nghị 8GB+)
- **Dung lượng**: ~2GB cho dependencies

## 🛠️ Bước 1: Chuẩn Bị Môi Trường

### Option A: Sử dụng Terminal/Command Prompt

```bash
# Kiểm tra Python version
python --version
# hoặc
python3 --version

# Nếu chưa có Python, tải từ: https://python.org/downloads/
```

### Option B: Sử dụng VS Code
1. Tải và cài đặt **VS Code**: https://code.visualstudio.com/
2. Cài extension **Python** từ Microsoft
3. Mở folder project trong VS Code

## 📂 Bước 2: Tải và Giải Nén Project

1. **Tải project** về máy (từ GitHub, email, hoặc USB)
2. **Giải nén** vào thư mục mong muốn
3. **Mở terminal** tại thư mục project:

```bash
# Windows (Command Prompt)
cd "đường_dẫn_tới_project\python_project_template"

# macOS/Linux (Terminal)
cd "đường_dẫn_tới_project/python_project_template"

# Hoặc trong VS Code: Terminal > New Terminal
```

## 🐍 Bước 3: Tạo Virtual Environment (Khuyến nghị)

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# Bạn sẽ thấy (venv) xuất hiện trước dòng lệnh
```

## 📦 Bước 4: Cài Đặt Dependencies

### Option A: Sử dụng Makefile (Đơn giản nhất)
```bash
# Cài đặt tất cả dependencies
make install

# Hoặc cài đặt cho development
make install-dev
```

### Option B: Sử dụng pip trực tiếp
```bash
# Cài đặt dependencies cơ bản
pip install -r requirements.txt

# Hoặc cài đặt cho development
pip install -r requirements-dev.txt
```

## 🚀 Bước 5: Chạy Ứng Dụng

### Option A: Chạy Streamlit App (Khuyến nghị)
```bash
# Chạy ứng dụng web
streamlit run src/stock_predictor/app.py
```

### Option B: Sử dụng Makefile
```bash
make run
```

### Option C: Chạy Main Module
```bash
# Chạy từ thư mục gốc
cd src
python -m stock_predictor.main

# Hoặc chạy trực tiếp (nếu import được fix)
python src/stock_predictor/main.py
```

## 🌐 Bước 6: Mở Ứng Dụng Trong Browser

Sau khi chạy lệnh, bạn sẽ thấy:

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

**Mở browser** và truy cập: **http://localhost:8501**

## 🎯 Bước 7: Sử Dụng Ứng Dụng

### 📊 **Demo Dữ Liệu Mẫu**
1. Chọn **"Demo Dữ Liệu Mẫu"** trong sidebar
2. Xem phân tích tự động với dữ liệu VN30
3. Thử **"🚀 Chạy Demo Huấn Luyện Mô Hình"**
4. Nhấn **"🧠 Nhận Dự Báo Thị Trường AI"** trong sidebar

### 📁 **Tải File CSV**
1. Chọn **"Tải File CSV"** trong sidebar
2. Upload file CSV với định dạng VN30 (như VN30_demo.csv)
3. Nhấn **"🔄 Xử Lý Dữ Liệu Đã Tải Lên"**
4. Xem phân tích và sử dụng các tính năng ML

### 🔮 **Demo Dự Báo**
1. Chọn **"Demo Dự Báo"** 
2. Chọn chỉ số (USD/VND hoặc Gold)
3. Chọn số ngày dự báo
4. Nhấn **"🔮 Tạo Dự Báo"**

## 📁 Định Dạng File CSV Được Hỗ Trợ

### Định dạng VN30 (Khuyến nghị):
```csv
Date;Close;Open;High;Low;Volumn;% turnover
01/08/2025;1,614.11;1,615.23;1,621.64;1,584.98;506.72M;-0.07%
01/07/2025;1,615.23;1,477.56;1,702.30;1,466.97;10.85B;9.32%
```

### Định dạng chuẩn quốc tế:
```csv
date,open,high,low,close,volume,turnover
2025-08-01,1615.23,1621.64,1584.98,1614.11,506720000,1614.11
```

## 🛠️ Các Lệnh Hữu Ích

```bash
# Dọn dẹp cache
make clean

# Chạy tests
make test

# Format code
make format

# Kiểm tra code quality
make lint

# Xem tất cả lệnh có sẵn
make help
```

## ❌ Xử Lý Lỗi Thường Gặp

### 🐍 **Lỗi Python không tìm thấy**
```bash
# Cài đặt Python từ python.org
# Hoặc sử dụng python3 thay vì python
python3 --version
```

### 📦 **Lỗi cài đặt packages**
```bash
# Cập nhật pip
pip install --upgrade pip

# Cài đặt lại requirements
pip install -r requirements.txt --force-reinstall
```

### 🌐 **Lỗi không mở được localhost**
```bash
# Thử port khác
streamlit run src/stock_predictor/app.py --server.port 8502

# Kiểm tra firewall
# Tắt VPN nếu có
```

### 🔒 **Lỗi quyền truy cập**
```bash
# Windows: Chạy Command Prompt với quyền Administrator
# macOS/Linux: Thêm sudo (nếu cần thiết)
sudo pip install -r requirements.txt
```

### 📦 **Lỗi Import Module**
```bash
# Lỗi: ImportError: attempted relative import with no known parent package

# Giải pháp 1: Chạy từ thư mục src
cd src
python -m stock_predictor.main

# Giải pháp 2: Thêm src vào PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"  # Linux/macOS
set PYTHONPATH=%PYTHONPATH%;%CD%\src          # Windows

# Giải pháp 3: Chỉ chạy Streamlit app (đơn giản nhất)
streamlit run src/stock_predictor/app.py
```

## 📚 Tài Liệu Thêm

- **README.md**: Thông tin chi tiết về project
- **docs/**: Documentation đầy đủ
- **examples/**: Ví dụ sử dụng
- **GitHub Issues**: Báo lỗi và hỗ trợ

## 🆘 Hỗ Trợ

Nếu gặp vấn đề:
1. **Kiểm tra** requirements.txt đã được cài đầy đủ
2. **Xem** error message trong terminal
3. **Thử** chạy từng bước một cách cẩn thận
4. **Liên hệ** developer để được hỗ trợ

---

## ✅ Checklist Hoàn Thành

- [ ] Python 3.8+ đã cài đặt
- [ ] Project đã được tải và giải nén
- [ ] Virtual environment đã được tạo và kích hoạt
- [ ] Dependencies đã được cài đặt thành công
- [ ] Ứng dụng chạy được trên localhost:8501
- [ ] Đã test các tính năng cơ bản

**🎉 Chúc bạn sử dụng thành công!**
