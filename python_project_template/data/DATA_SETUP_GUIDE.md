# 📊 Hướng Dẫn Thêm Dữ Liệu Cho Demo Forecast

## ❌ Lỗi: "Không thể tải dữ liệu từ Desktop"

User gặp lỗi này vì ứng dụng không tìm thấy 2 file dữ liệu cần thiết cho chức năng Demo Forecast.

## 📁 Đường Dẫn File Cần Thiết

Ứng dụng đang tìm kiếm 2 file sau trên Desktop:

```
/Users/dungnhi/Desktop/Dữ liệu Lịch sử USD_VND.csv
/Users/dungnhi/Desktop/dữ liệu lịch sử giá vàng.csv
```

## 🔧 Cách Giải Quyết

### Cách 1: Tạo File Dữ Liệu Mẫu (Khuyến nghị)

1. **Tạo file USD_VND.csv** trong thư mục `data/`:
   ```
   python_project_template/data/USD_VND_sample.csv
   ```

2. **Tạo file Gold.csv** trong thư mục `data/`:
   ```
   python_project_template/data/Gold_sample.csv
   ```

### Cách 2: Cung Cấp File Thật

Nếu bạn có dữ liệu thật, đặt 2 file sau vào Desktop:
- `Dữ liệu Lịch sử USD_VND.csv`
- `dữ liệu lịch sử giá vàng.csv`

### Cách 3: Sửa Đường Dẫn Trong Code

Chỉnh sửa file `src/stock_predictor/forecast/forecaster.py` dòng 29-31:

```python
# Thay vì:
desktop_path = "/Users/dungnhi/Desktop"
usd_file = os.path.join(desktop_path, "Dữ liệu Lịch sử USD_VND.csv")
gold_file = os.path.join(desktop_path, "dữ liệu lịch sử giá vàng.csv")

# Thành:
data_path = "data"  # hoặc đường dẫn tới thư mục dữ liệu của bạn
usd_file = os.path.join(data_path, "USD_VND_sample.csv")
gold_file = os.path.join(data_path, "Gold_sample.csv")
```

## 📊 Format Dữ Liệu Cần Thiết

Các file CSV cần có format như sau:

### USD_VND.csv:
```csv
Date,Open,High,Low,Close,Volume
2023-01-01,23800,23850,23750,23820,1000000
2023-01-02,23820,23870,23800,23845,1200000
...
```

### Gold.csv:
```csv
Date,Open,High,Low,Close,Volume
2023-01-01,1800.50,1805.20,1798.30,1802.75,50000
2023-01-02,1802.75,1808.90,1800.15,1806.40,55000
...
```

## 🎯 Giải Pháp Nhanh

Để tạm thời bỏ qua lỗi này và vẫn sử dụng ứng dụng:

1. **Sử dụng VN30 Demo** thay vì USD/Gold Demo
2. **Upload CSV** của riêng bạn
3. **Bỏ qua phần Forecast Demo** và sử dụng các tính năng khác

## 📞 Hỗ Trợ

Nếu vẫn gặp lỗi, vui lòng:
1. Kiểm tra đường dẫn file
2. Đảm bảo format CSV đúng
3. Kiểm tra quyền truy cập file
4. Liên hệ để được hỗ trợ thêm
