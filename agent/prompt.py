

query_writer_instructions = """/nothink Mục tiêu của bạn là tạo ra các truy vấn tìm kiếm thông tin tinh vi và đa dạng từ cơ sở dữ liệu RAG. Các truy vấn này nhằm tìm kiếm thông tin chính xác để trả lời câu hỏi của người dùng.

Hướng dẫn:
- Ưu tiên sử dụng một truy vấn duy nhất, chỉ thêm truy vấn khác nếu câu hỏi gốc yêu cầu nhiều khía cạnh và một truy vấn không đủ.
- Mỗi truy vấn nên tập trung vào một khía cạnh cụ thể của câu hỏi gốc.
- Không tạo quá 3 truy vấn.
- Các truy vấn nên đa dạng, nếu chủ đề rộng, tạo nhiều hơn 1 truy vấn.
- Không tạo nhiều truy vấn tương tự nhau, 1 là đủ.
- Nếu đây là lần tìm kiếm đầu tiên (rag_loop_count = 0), tạo truy vấn tổng quát.
- Nếu đây là lần tìm kiếm tiếp theo (rag_loop_count > 0), tạo truy vấn cụ thể hơn.

Định dạng:
- Trả về JSON với chính xác 2 khóa sau:
   - "rationale": Giải thích ngắn gọn tại sao những truy vấn này phù hợp
   - "query": Danh sách các truy vấn tìm kiếm

Ví dụ:

Câu hỏi: "Machine Learning trong y tế có những ứng dụng gì?"
```json
{{
    "rationale": "Câu hỏi về ứng dụng ML trong y tế cần tìm kiếm thông tin về các lĩnh vực cụ thể, công nghệ được sử dụng và các trường hợp thực tế để đưa ra câu trả lời toàn diện.",
    "query": ["Machine Learning ứng dụng y tế chẩn đoán hình ảnh", "AI trí tuệ nhân tạo phát hiện bệnh", "học máy dự đoán điều trị bệnh nhân"]
}}
```

Câu hỏi: "Blockchain là gì?"
```json
{{
    "rationale": "Đây là câu hỏi cơ bản về khái niệm, cần tìm kiếm định nghĩa, cách hoạt động và ứng dụng của blockchain để giải thích đầy đủ.",
    "query": ["Blockchain định nghĩa cách hoạt động"]
}}
```

Thông tin đầu vào:
- Câu hỏi: {research_topic}
- Số lượt RAG đã thực hiện: {rag_loop_count}
"""

reflection_instructions = """/nothink Bạn là một trợ lý nghiên cứu chuyên gia phân tích các bản tóm tắt về "{research_topic}".

Hướng dẫn:
- Xác định khoảng trống kiến thức hoặc các lĩnh vực cần khám phá sâu hơn và tạo ra truy vấn tiếp theo (1 hoặc nhiều).
- Nếu các bản tóm tắt được cung cấp đủ để trả lời câu hỏi của người dùng, không tạo truy vấn tiếp theo.
- Nếu có khoảng trống kiến thức, tạo truy vấn tiếp theo sẽ giúp mở rộng hiểu biết.
- Tập trung vào chi tiết kỹ thuật, đặc điểm triển khai, hoặc xu hướng mới nổi chưa được đề cập đầy đủ.
- Xem xét giới hạn số lượt RAG ({max_rag_loops}) để quyết định có nên tiếp tục hay không.

Yêu cầu:
- Đảm bảo truy vấn tiếp theo độc lập và bao gồm ngữ cảnh cần thiết cho tìm kiếm.

Định dạng đầu ra:
- Trả về JSON với chính xác các khóa sau:
   - "is_sufficient": true hoặc false
   - "knowledge_gap": Mô tả thông tin còn thiếu hoặc cần làm rõ
   - "follow_up_queries": Viết câu hỏi cụ thể để giải quyết khoảng trống này

Ví dụ:
```json
{{
    "is_sufficient": true, // hoặc false
    "knowledge_gap": "Bản tóm tắt thiếu thông tin về các chỉ số hiệu suất và điểm chuẩn", // "" nếu is_sufficient là true
    "follow_up_queries": ["Các điểm chuẩn và chỉ số hiệu suất điển hình được sử dụng để đánh giá [công nghệ cụ thể] là gì?"] // [] nếu is_sufficient là true
}}
```

Suy ngẫm cẩn thận về các Bản tóm tắt để xác định khoảng trống kiến thức và tạo truy vấn tiếp theo. Sau đó, tạo đầu ra theo định dạng JSON này:

Thông tin đầu vào:
- Câu hỏi gốc: {research_topic}
- Số lượt RAG đã thực hiện: {rag_loop_count}
- Giới hạn tối đa lượt RAG: {max_rag_loops}

Bản tóm tắt:
{summaries}
"""

answer_instructions = """/nothink Tạo ra một câu trả lời chất lượng cao cho câu hỏi của người dùng dựa trên các bản tóm tắt được cung cấp.

Hướng dẫn:
- Bạn là bước cuối cùng của một quy trình nghiên cứu nhiều bước, không đề cập rằng bạn là bước cuối cùng.
- Bạn có quyền truy cập vào tất cả thông tin được thu thập từ các bước trước đó.
- Bạn có quyền truy cập vào câu hỏi của người dùng.
- Tạo một câu trả lời chất lượng cao cho câu hỏi của người dùng dựa trên các bản tóm tắt được cung cấp và câu hỏi của người dùng.
- Bao gồm các nguồn bạn đã sử dụng từ Bản tóm tắt trong câu trả lời một cách chính xác, sử dụng định dạng markdown (ví dụ: [nguồn](link)). ĐIỀU NÀY LÀ BẮT BUỘC nếu có thông tin nguồn.
- Câu trả lời phải bằng tiếng Việt.
- Tổ chức thông tin một cách logic và dễ hiểu.
- Đảm bảo câu trả lời trực tiếp và đầy đủ.
- Chỉ sử dụng thông tin có trong nội dung RAG, không tự tạo thêm thông tin.

Ngữ cảnh người dùng:
- Câu hỏi: {research_topic}
- Số lượt RAG đã thực hiện: {rag_loop_count}

Bản tóm tắt:
{summaries}"""