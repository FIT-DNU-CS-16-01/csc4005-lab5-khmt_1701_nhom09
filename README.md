[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kUEG02mW)
# CSC4005 Lab 5 – Vision Transformer for Smart Campus Scene Classification

Repo này là phần triển khai hoàn chỉnh cho Lab 5 của học phần CSC4005: phân loại ảnh ngữ cảnh Smart Campus bằng Vision Transformer trên subset 5 lớp của MIT Indoor Scenes 67.

## Thông tin nhóm

| Họ tên | Mã sinh viên | Lớp |
|---|---|---|
| Lưu Thanh Tùng | 1771040029 | KHMT-1701 |
| Nguyễn Hoàng Anh | 1771040002 | KHMT-1701 |

## Tóm tắt bài toán

Mô hình cần phân loại ảnh vào 5 lớp gần với bối cảnh đại học:

```text
classroom
computerroom
library
corridor
office
```

Trong repo này, nhóm đã triển khai và so sánh các cấu hình sau:

- ViT-B/16 `head_only`
- ViT-B/16 `finetune`
- ViT-B/16 `head_only` không augmentation
- ViT-B/16 `finetune` với learning rate thấp hơn
- ResNet18 `head_only` làm mô hình đối chứng

## Kết quả chính

Run tốt nhất của nhóm là `1771040029_vit_finetune_e50`.

| Chỉ số | Giá trị |
|---|---:|
| Validation accuracy tốt nhất | 98.28% |
| Validation macro-F1 tốt nhất | 0.9754 |
| Test accuracy | 95.69% |
| Test macro-F1 | 0.9465 |
| Best epoch | 17 |

So sánh các run chính:

| Run | Mô hình | Train mode | Test acc | Test macro-F1 |
|---|---|---|---:|---:|
| `vit_b16_head_only` | ViT-B/16 | `head_only` | 93.97% | 0.9193 |
| `1771040029_vit_finetune_e50` | ViT-B/16 | `finetune` | 95.69% | 0.9465 |
| `1771040029_vit_head_only_noaug_e50` | ViT-B/16 | `head_only` | 93.10% | 0.9072 |
| `1771040029_vit_finetune_lowlr_e50` | ViT-B/16 | `finetune` | 95.69% | 0.9433 |
| `1771040029_resnet18_head_only_e50` | ResNet18 | `head_only` | 87.93% | 0.8424 |

Kết luận nhanh:

- `finetune` tốt hơn `head_only` trên cùng backbone ViT.
- Augmentation giúp bản `head_only` tốt hơn khoảng 0.0122 macro-F1.
- Giảm learning rate trong `finetune` không cải thiện test so với cấu hình chuẩn.
- CNN chạy nhanh hơn nhưng thua ViT khá rõ trên bài toán scene classification.

## Dữ liệu sử dụng

Nhóm chạy trên 5 lớp trong `data/indoorCVPR_09/Images`.

| Lớp | Số ảnh |
|---|---:|
| classroom | 113 |
| computerroom | 114 |
| library | 107 |
| corridor | 346 |
| office | 109 |
| Tổng | 789 |

Chia dữ liệu theo tỉ lệ gần `70/15/15`, tương ứng:

| Tập | Số ảnh |
|---|---:|
| Train | 557 |
| Validation | 116 |
| Test | 116 |

Repo không commit dữ liệu ảnh gốc.

## Thiết lập môi trường

Theo yêu cầu làm việc của nhóm, toàn bộ thí nghiệm được chạy trong môi trường:

```powershell
conda activate HocSau
pip install -r requirements.txt
```

Huấn luyện chính yêu cầu GPU CUDA. `src/train.py` hiện mặc định ưu tiên `cuda`.

## Lệnh chạy

Kiểm tra pipeline nhanh:

```powershell
conda activate HocSau
python -m src.train --config configs/debug_smoke.json --data_dir data/indoorCVPR_09/Images --run_name debug_smoke_gpu_check
```

Baseline `head_only`:

```powershell
conda activate HocSau
python -m src.train --config configs/baseline_vit_head_only.json --data_dir data/indoorCVPR_09/Images --run_name 1771040029_vit_head_only_e50
```

Run tốt nhất `finetune`:

```powershell
conda activate HocSau
python -m src.train --config configs/baseline_vit_finetune.json --data_dir data/indoorCVPR_09/Images --run_name 1771040029_vit_finetune_e50
```

Run so sánh không augmentation:

```powershell
conda activate HocSau
python -m src.train --config configs/baseline_vit_head_only_noaug.json --data_dir data/indoorCVPR_09/Images --run_name 1771040029_vit_head_only_noaug_e50
```

Run `finetune` với learning rate thấp:

```powershell
conda activate HocSau
python -m src.train --config configs/baseline_vit_finetune_lowlr.json --data_dir data/indoorCVPR_09/Images --run_name 1771040029_vit_finetune_lowlr_e50
```

Run CNN đối chứng:

```powershell
conda activate HocSau
python -m src.train --config configs/baseline_resnet18_head_only.json --data_dir data/indoorCVPR_09/Images --run_name 1771040029_resnet18_head_only_e50
```

## W&B

- Project: https://wandb.ai/thanhtung-contact-official-/csc4005-lab6-mit-indoor-vit
- Run tốt nhất: https://wandb.ai/thanhtung-contact-official-/csc4005-lab6-mit-indoor-vit/runs/giytbsjq
- Run `head_only` không augmentation: https://wandb.ai/thanhtung-contact-official-/csc4005-lab6-mit-indoor-vit/runs/pk93asgd
- Run `finetune` low learning rate: https://wandb.ai/thanhtung-contact-official-/csc4005-lab6-mit-indoor-vit/runs/8of78m19
- Run CNN đối chứng: https://wandb.ai/thanhtung-contact-official-/csc4005-lab6-mit-indoor-vit/runs/v7mjy1su

## Output quan trọng

Mỗi run lưu trong `outputs/<run_name>/` với các file:

```text
best_model.pt
history.csv
metrics.json
curves.png
confusion_matrix.png
class_to_idx.json
config.json
```

Artifacts nên xem đầu tiên:

- `outputs/1771040029_vit_finetune_e50/metrics.json`
- `outputs/1771040029_vit_finetune_e50/curves.png`
- `outputs/1771040029_vit_finetune_e50/confusion_matrix.png`
- `outputs/1771040029_resnet18_head_only_e50/metrics.json`

Lưu ý: baseline `head_only` 50 epoch đầu tiên được lưu dưới `outputs/vit_b16_head_only/` do lỗi override config trước khi sửa. Kết quả vẫn hợp lệ và đã được ghi nhận trong tài liệu.

## Báo cáo và tài liệu liên quan

- Báo cáo đã điền: `REPORT_TEMPLATE.md`
- Kế hoạch và checklist: `docs/PLAN.md`
- Tóm tắt kết quả triển khai: `docs/Lab5.md`
- Hướng dẫn dataset: `docs/DATASET_GUIDE.md`
- Hướng dẫn W&B: `docs/WANDB_GUIDE.md`

## Nhận xét cuối cùng

Trên dataset hiện tại, ViT phù hợp hơn CNN cho bài toán scene classification vì mô hình tận dụng ngữ cảnh toàn cục tốt hơn. Tuy nhiên, cái giá phải trả là thời gian train cao hơn và chi phí fine-tune lớn hơn. Với yêu cầu nộp bài hiện tại, `1771040029_vit_finetune_e50` là lựa chọn tốt nhất để đưa vào báo cáo chính, còn `resnet18` phù hợp làm thí nghiệm mở rộng để phân tích sự khác nhau giữa CNN và ViT.
