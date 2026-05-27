# Lab 5 - Smart Campus Scene Classification

## Tổng quan triển khai của nhóm

Tài liệu này tóm tắt phần triển khai thực tế của nhóm cho Lab 5, dựa trên kết quả đã train xong trong repo. Nội dung này không thay thế rubric hay hướng dẫn chi tiết trong các file khác, mà đóng vai trò bản tóm tắt kỹ thuật để đối chiếu nhanh kết quả, cấu hình và kết luận chính.

## Thông tin nhóm

| Họ tên | Mã sinh viên | Lớp |
|---|---|---|
| Lưu Thanh Tùng | 1771040029 | KHMT-1701 |
| Nguyễn Hoàng Anh | 1771040002 | KHMT-1701 |

## Mục tiêu đã thực hiện

Nhóm đã hoàn thành các mục tiêu chính của Lab 5:

- chuẩn bị và sử dụng subset 5 lớp của MIT Indoor Scenes 67
- chạy được ViT-B/16 ở chế độ `head_only`
- chạy được ViT-B/16 ở chế độ `finetune`
- log đầy đủ các run chính bằng W&B
- lưu `metrics.json`, `history.csv`, `curves.png`, `confusion_matrix.png`, `class_to_idx.json`
- thực hiện các run mở rộng để so sánh augmentation, learning rate và CNN đối chứng

## Dữ liệu sử dụng

Subset gồm 5 lớp:

```text
classroom
computerroom
library
corridor
office
```

Thống kê dữ liệu:

| Lớp | Số ảnh |
|---|---:|
| classroom | 113 |
| computerroom | 114 |
| library | 107 |
| corridor | 346 |
| office | 109 |
| Tổng | 789 |

Chia tập dữ liệu:

| Tập | Số ảnh |
|---|---:|
| Train | 557 |
| Validation | 116 |
| Test | 116 |

Nhận xét: lớp `corridor` có số lượng ảnh lớn hơn hẳn các lớp còn lại, nên dữ liệu có mất cân bằng tương đối rõ.

## Cấu hình chính của nhóm

Run tốt nhất của nhóm là `1771040029_vit_finetune_e50` với cấu hình:

| Thành phần | Giá trị |
|---|---|
| model_name | `vit_b_16` |
| train_mode | `finetune` |
| epochs | 50 |
| batch_size | 8 |
| lr | 0.00005 |
| img_size | 224 |
| dropout | 0.2 |
| optimizer | AdamW |
| weight_decay | 0.0001 |
| device | `cuda` |
| total_params | 85,802,501 |
| trainable_params | 85,802,501 |
| trainable_ratio | 1.0 |

## Kết quả các run đã thực hiện

| Run | Mô hình | Train mode | Augment | LR | Test acc | Test macro-F1 | Ghi chú |
|---|---|---|---|---:|---:|---:|---|
| `vit_b16_head_only` | ViT-B/16 | `head_only` | Có | 0.001 | 93.97% | 0.9193 | baseline đầu tiên, lưu dưới tên cũ do lỗi override config trước khi sửa |
| `1771040029_vit_finetune_e50` | ViT-B/16 | `finetune` | Có | 0.00005 | 95.69% | 0.9465 | run tốt nhất hiện tại |
| `1771040029_vit_head_only_noaug_e50` | ViT-B/16 | `head_only` | Không | 0.001 | 93.10% | 0.9072 | dùng để so sánh augmentation |
| `1771040029_vit_finetune_lowlr_e50` | ViT-B/16 | `finetune` | Có | 0.00002 | 95.69% | 0.9433 | val tốt nhưng test thấp hơn cấu hình finetune chuẩn |
| `1771040029_resnet18_head_only_e50` | ResNet18 | `head_only` | Có | 0.001 | 87.93% | 0.8424 | CNN đối chứng |

## Kết luận thực nghiệm

### 1. So sánh `head_only` và `finetune`

- `finetune` cho kết quả tốt hơn `head_only` trên cả accuracy và macro-F1.
- Run tốt nhất `1771040029_vit_finetune_e50` đạt `test_macro_f1 = 0.9465`, cao hơn baseline `head_only` khoảng 0.0272.
- `best_epoch` của run tốt nhất là 17, nghĩa là dù huấn luyện 50 epoch, checkpoint tốt nhất xuất hiện sớm hơn khá nhiều.

### 2. Tác động của augmentation

- Với `head_only`, augmentation giúp tăng `test_macro_f1` từ 0.9072 lên 0.9193.
- Điều này cho thấy augmentation nhẹ vẫn có ích khi chỉ train classification head trên dataset không quá lớn.

### 3. Tác động của learning rate thấp hơn

- Cấu hình `finetune` với `lr = 0.00002` đạt `best_val_macro_f1` hơi cao hơn rất nhẹ, nhưng `test_macro_f1` thấp hơn bản chuẩn `lr = 0.00005`.
- Kết quả này cho thấy learning rate thấp hơn chưa chắc mang lại lợi ích thực tế trên tập test.

### 4. So sánh ViT với CNN

- ResNet18 nhanh hơn nhiều nhưng kém ViT rõ rệt trên bài toán này.
- Thời gian epoch trung bình:

| Run | Thời gian epoch trung bình |
|---|---:|
| `vit_b16_head_only` | 23.36 giây |
| `1771040029_vit_finetune_e50` | 32.51 giây |
| `1771040029_resnet18_head_only_e50` | 8.14 giây |

- ViT phù hợp hơn cho scene classification vì mô hình học quan hệ toàn cục giữa các patch tốt hơn CNN đối với ảnh ngữ cảnh trong nhà.

## Phân tích lỗi từ confusion matrix

Từ confusion matrix của run tốt nhất `1771040029_vit_finetune_e50`:

- `classroom`, `library` và `corridor` được dự đoán đúng hoàn toàn trên test set.
- `computerroom` bị nhầm sang `office` 3 ảnh.
- `office` bị nhầm sang `corridor` 2 ảnh.

Giải thích hợp lý:

- `computerroom` và `office` cùng là không gian trong nhà có bàn ghế, máy tính và bố cục làm việc, nên dễ có đặc trưng thị giác gần nhau.
- Một số ảnh `office` có không gian mở hoặc góc chụp gần lối đi, nên bị nhầm sang `corridor`.

Từ confusion matrix của `resnet18`:

- CNN nhầm `classroom` sang `computerroom` và `office` nhiều hơn ViT.
- `library` cũng bị nhầm sang `corridor`.
- Điều này gợi ý rằng ViT tận dụng ngữ cảnh toàn cục tốt hơn CNN trong bài toán phân loại cảnh.

## Bằng chứng W&B

- Project: https://wandb.ai/thanhtung-contact-official-/csc4005-lab6-mit-indoor-vit
- Run tốt nhất: https://wandb.ai/thanhtung-contact-official-/csc4005-lab6-mit-indoor-vit/runs/giytbsjq
- Run `head_only` không augmentation: https://wandb.ai/thanhtung-contact-official-/csc4005-lab6-mit-indoor-vit/runs/pk93asgd
- Run `finetune` low learning rate: https://wandb.ai/thanhtung-contact-official-/csc4005-lab6-mit-indoor-vit/runs/8of78m19
- Run CNN đối chứng: https://wandb.ai/thanhtung-contact-official-/csc4005-lab6-mit-indoor-vit/runs/v7mjy1su

## Artefact quan trọng trong repo

- `outputs/vit_b16_head_only/`
- `outputs/1771040029_vit_finetune_e50/`
- `outputs/1771040029_vit_head_only_noaug_e50/`
- `outputs/1771040029_vit_finetune_lowlr_e50/`
- `outputs/1771040029_resnet18_head_only_e50/`

Các file cần xem khi viết báo cáo:

- `metrics.json`
- `history.csv`
- `curves.png`
- `confusion_matrix.png`
- `config.json`

## Kết luận cuối cùng

Với bài toán phân loại ảnh Smart Campus trên subset 5 lớp của MIT Indoor Scenes 67, mô hình ViT-B/16 `finetune` là lựa chọn tốt nhất trong các cấu hình đã thử. Augmentation có lợi cho `head_only`, còn learning rate thấp hơn trong `finetune` không đem lại cải thiện test rõ rệt. CNN có thể dùng làm đối chứng vì nhanh hơn, nhưng hiện chưa cạnh tranh được với ViT về chất lượng trên bài toán này.