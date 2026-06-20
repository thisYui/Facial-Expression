# Đồ án môn học Nhận dạng mẫu: Facial Expression Recognition

## 1. Giới thiệu

Đây là đồ án môn học **Nhận dạng mẫu**, tập trung vào chủ đề **Facial Expression Recognition**. Nội dung đồ án bao gồm nghiên cứu lý thuyết từ giáo trình, khảo sát bài báo tiên tiến, phân tích mã nguồn, chạy thực nghiệm và demo minh họa nhận diện cảm xúc khuôn mặt.

Nhóm nghiên cứu chương **Facial Expression** từ giáo trình **Handbook of Biometrics hoặc Handbook of Face Recognition 2nd Edition**. Bên cạnh đó, nhóm chọn bài báo SOTA **Generalizable Facial Expression Recognition**

## 2. Cấu trúc thư mục

```text
Facial Expression/
├── demo/
│   └── link.txt
│
├── papers/
│   └── Generalizable Facial Expression Recognition.pdf
│
├── reports/
│   ├── appendix/
│   ├── content/
│   ├── images/
│   ├── ref/
│   ├── main.tex
│   └── Report.pdf
│
├── slide/
│   └── Presentation.pdf
│
├── source/
│   ├── code/
│   ├── cross_dataset_evaluations/
│   ├── notebooks/
│   ├── result/
│   ├── scripts/
│   ├── README.md
│   └── requirements.txt
│
└── README.md
```

Trong đó:

* `papers/`: chứa bài báo khoa học được nhóm lựa chọn để nghiên cứu.
* `reports/`: chứa mã nguồn LaTeX và file báo cáo hoàn chỉnh.
* `slide/`: chứa slide trình bày của nhóm.
* `demo/`: chứa liên kết Google Drive video demo.
* `source/`: chứa mã nguồn, notebook, script xử lý, kết quả thực nghiệm và các file cấu hình cần thiết.

## 3. Phân công nhiệm vụ

| MSSV     | Thành viên         | Nhiệm vụ phụ trách                                                                                                                                           | Tỷ lệ hoàn thành |
| -------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------- |
| 22120134 | Hoàng Tiến Huy     | Tìm hiểu mã nguồn của phương pháp SOTA, phân tích cơ sở thiết kế của hệ thống và triển khai demo nhận diện cảm xúc theo thời gian thực bằng webcam.          | 100%             |
| 23120195 | Lê Hà Thanh Chương | Khảo sát một bài báo SOTA về nhận diện cảm xúc, phân tích động lực nghiên cứu, ý tưởng cốt lõi, điểm khác biệt so với giáo trình và đưa ra nhận xét cá nhân. | 100%             |
| 23120232 | Lê Thượng Đế       | Nghiên cứu nội dung Facial Expression trong giáo trình, hệ thống hóa nền tảng lý thuyết, quy trình xử lý và các công thức toán học liên quan.                | 100%             |
| 23120245 | Nguyễn Quang Duy   | Tổng hợp nội dung và kết quả của nhóm, rà soát tính thống nhất của báo cáo, xây dựng slide trình bày và chuẩn bị kịch bản thuyết trình.                      | 100%             |
| 23120258 | Lưu Trọng Hiếu     | Ghi nhận kết quả thực nghiệm từ mã nguồn, đối chiếu với kết quả trong bài báo, phân tích nguyên nhân sai lệch và hoàn thiện phần nhận xét thực nghiệm.       | 100%             |
| 23120260 | Văn Đình Hiếu      | Cài đặt môi trường thực nghiệm, chạy lại mã nguồn của bài báo và thực hiện thêm thử nghiệm trên dataset ngoài để đánh giá khả năng tổng quát của mô hình.    | 100%             |

## 4. Tài liệu tham khảo chính

* **Handbook of Biometrics hoặc Handbook of Face Recognition 2nd Edition**
* **Generalizable Facial Expression Recognition**