# Agentic RAG Agent - Web Interface

Một giao diện web hiện đại và đẹp mắt cho Agentic RAG Agent, được xây dựng với FastAPI và HTML/CSS/JavaScript.

## 🌟 Tính năng

- **Giao diện chat hiện đại**: Thiết kế responsive với hiệu ứng mượt mà
- **Real-time conversation**: Chat trực tiếp với RAG agent
- **Hiển thị nguồn tài liệu**: Xem các file được tham khảo
- **Thống kê chi tiết**: Số vòng RAG, truy vấn, nguồn tài liệu
- **Lịch sử chat**: Lưu trữ và quản lý cuộc hội thoại
- **API RESTful**: Backend API đầy đủ với documentation
- **Responsive design**: Tương thích với mọi thiết bị

## 🚀 Cài đặt và Chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Khởi động web server

**Windows:**
```cmd
start_web.bat
```

**Linux/MacOS:**
```bash
chmod +x start_web.sh
./start_web.sh
```

**Hoặc chạy trực tiếp:**
```bash
python app.py
```

### 3. Truy cập ứng dụng

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## 📱 Giao diện

### Chat Interface
- Thiết kế hiện đại với gradient màu sắc
- Avatar cho user và assistant
- Hiển thị thời gian thực và thông tin xử lý
- Auto-scroll khi có tin nhắn mới
- Support cho multi-line messages

### Features
- **Send Message**: Enter để gửi, Shift+Enter để xuống dòng
- **Clear Chat**: Xóa toàn bộ lịch sử
- **Real-time Status**: Hiển thị trạng thái online/offline
- **Loading Animation**: Hiệu ứng khi agent đang xử lý
- **Source Display**: Hiển thị các file tài liệu được tham khảo
- **Statistics**: Thống kê về quá trình RAG

## 🔧 API Endpoints

### POST /api/chat
Chat với RAG agent

**Request:**
```json
{
    "message": "Câu hỏi của bạn",
    "config": {
        "max_rag_loops": 3,
        "number_of_initial_queries": 2
    }
}
```

**Response:**
```json
{
    "response": "Câu trả lời từ agent",
    "sources": ["file1.txt", "file2.pdf"],
    "stats": {
        "rag_loop_count": 2,
        "total_queries": 4,
        "unique_sources": 2
    },
    "timestamp": "2024-01-01T12:00:00"
}
```

### GET /api/chat/history
Lấy lịch sử chat

### DELETE /api/chat/history  
Xóa lịch sử chat

### GET /api/health
Kiểm tra tình trạng hệ thống

## 🎨 Thiết kế

### Color Scheme
- **Primary**: Gradient xanh dương đến tím (#4f46e5 → #7c3aed)
- **Background**: Gradient xanh nhạt (#667eea → #764ba2)  
- **User Messages**: Gradient xanh dương (#3b82f6 → #1d4ed8)
- **Assistant**: Gradient xanh lá (#10b981 → #059669)

### Typography
- **Font**: Segoe UI, modern sans-serif
- **Responsive**: Tự động điều chỉnh theo màn hình

### Animations
- **Fade-in**: Tin nhắn mới xuất hiện mượt mà
- **Pulse**: Trạng thái online
- **Bounce**: Loading dots
- **Hover effects**: Button interactions

## 🔧 Cấu hình

Bạn có thể điều chỉnh cấu hình trong file `app.py`:

```python
default_config = {
    "query_generator_model": "qwen/qwen3-4b",
    "reflection_model": "qwen/qwen3-4b", 
    "rag_model": "qwen/qwen3-4b",
    "answer_model": "qwen/qwen3-4b",
    "max_rag_loops": 3,
    "number_of_initial_queries": 2
}
```

## 📁 Cấu trúc Project

```
agentic_rag/
├── app.py              # FastAPI web server
├── static/
│   └── index.html      # Web interface
├── requirements.txt    # Dependencies
├── start_web.bat      # Windows start script
├── start_web.sh       # Linux/Mac start script
├── agent/             # RAG agent core
├── core/              # Model utilities  
├── rag/               # RAG processing
└── documents/         # Document storage
```

## 🐛 Troubleshooting

### Lỗi thường gặp:

1. **Import Error**: Chưa cài đặt dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. **Port đã được sử dụng**: Thay đổi port trong `app.py`
   ```python
   uvicorn.run("app:app", port=8001)
   ```

3. **Agent không phản hồi**: Kiểm tra kết nối LM Studio
   - Đảm bảo LM Studio đang chạy trên localhost:1234
   - Kiểm tra model đã được load

## 📝 Tùy chỉnh

### Thêm tính năng mới:
1. Thêm endpoint mới trong `app.py`
2. Cập nhật giao diện trong `static/index.html`
3. Thêm CSS/JavaScript tương ứng

### Thay đổi giao diện:
- Chỉnh sửa CSS trong `static/index.html`
- Thay đổi color scheme, fonts, layout
- Thêm animations hoặc effects

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch  
5. Create Pull Request

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết.

---

**Phát triển bởi**: Agentic RAG Team  
**Phiên bản**: 1.0.0  
**Ngày cập nhật**: 2024
