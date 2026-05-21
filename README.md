[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kUEG02mW)
# CSC4005 Lab 5 – Vision Transformer for Smart Campus Scene Classification

Starter kit này dành cho **Lab 5 của học phần CSC4005**.


Starter kit được thiết kế theo format của repo `csc4005_lab2_neu_cnn_starter`:

```text
csc4005_lab6_mit_indoor_vit_starter/
├── README.md
├── REPORT_TEMPLATE.md
├── requirements.txt
├── configs/
│   ├── baseline_vit_head_only.json
│   └── debug_smoke.json
├── docs/
│   ├── DATASET_GUIDE.md
│   ├── RUBRIC_LAB5.md
│   ├── LAB_GUIDE_LAB5.md
│   └── WANDB_GUIDE.md
├── notebooks/
│   └── lab5_demo.ipynb
├── outputs/
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── prepare_subset.py
│   ├── train.py
│   └── utils.py
├── ci/
│   ├── check_structure.py
│   └── smoke_train.py
└── .github/
    └── workflows/
        └── ci.yml
```

## 1. Case study

**Smart Campus Scene Classification with Vision Transformer**

Hệ thống Smart Campus nhận ảnh từ camera hoặc thiết bị quan sát trong trường. Mô hình cần phân loại ảnh vào một trong các loại không gian gần với môi trường đại học:

```text
classroom
computerroom
library
corridor
office
```

Bộ dữ liệu được sử dụng là **MIT Indoor Scenes 67**. Repo này không chứa dữ liệu gốc. Sinh viên cần tải dữ liệu từ nguồn chính thức và truyền đường dẫn qua `--data_dir`.

## 2. Vì sao dùng ViT?

Lab này giúp sinh viên nối trực tiếp lý thuyết ViT với thực hành:

```text
image → patches → patch embedding → transformer encoder → classification head
```

Sinh viên cần hiểu:

1. ảnh có thể được xem như một chuỗi các patch;
2. patch embedding trong ViT tương tự token embedding trong NLP;
3. positional embedding giúp mô hình giữ thông tin vị trí;
4. pretrained ViT có thể được fine-tune cho bài toán scene classification;
5. W&B được dùng để theo dõi và so sánh thí nghiệm.

## 3. Mục tiêu

Sinh viên cần:

1. chuẩn bị subset 5 lớp từ MIT Indoor Scenes 67;
2. chạy được mô hình **Vision Transformer** ở chế độ `head_only`;
3. tùy chọn chạy `finetune` nếu máy đủ mạnh;
4. log thí nghiệm bằng **Weights & Biases (W&B)**;
5. đánh giá bằng accuracy, macro-F1, confusion matrix;
6. phân tích lỗi và đề xuất cải thiện mô hình/dữ liệu.

## 4. Cài đặt

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Chuẩn bị dữ liệu

Repo này **không chứa thư mục ảnh**. Sinh viên truyền dữ liệu ngoài repo qua `--data_dir`.

Hỗ trợ:

1. thư mục MIT Indoor Scenes 67 đã giải nén, có các thư mục lớp;
2. subset 5 lớp đã chuẩn bị sẵn;
3. file ZIP chứa cấu trúc thư mục theo lớp.

Ví dụ cấu trúc subset:

```text
mit_indoor_smartcampus_5/
├── classroom/
├── computerroom/
├── library/
├── corridor/
└── office/
```

Có thể dùng script hỗ trợ:

```bash
python -m src.prepare_subset \
  --source_dir /duong_dan/indoorCVPR_09/Images \
  --output_dir /duong_dan/mit_indoor_smartcampus_5 \
  --classes classroom computerroom library corridor office \
  --max_per_class 400
```

## 6. Chạy baseline ViT head-only

```bash
python -m src.train \
  --data_dir /duong_dan/mit_indoor_smartcampus_5 \
  --project csc4005-lab6-mit-indoor-vit \
  --run_name vit_b16_head_only \
  --model_name vit_b_16 \
  --train_mode head_only \
  --epochs 10 \
  --batch_size 16 \
  --img_size 224 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --dropout 0.2 \
  --augment \
  --use_wandb
```

## 7. Tùy chọn: fine-tune toàn bộ ViT

Nếu máy đủ mạnh:

```bash
python -m src.train \
  --data_dir /duong_dan/mit_indoor_smartcampus_5 \
  --project csc4005-lab6-mit-indoor-vit \
  --run_name vit_b16_finetune \
  --model_name vit_b_16 \
  --train_mode finetune \
  --epochs 5 \
  --batch_size 8 \
  --img_size 224 \
  --lr 0.00005 \
  --weight_decay 0.0001 \
  --dropout 0.2 \
  --augment \
  --use_wandb
```

## 8. Chạy nhanh để kiểm tra pipeline

```bash
python -m src.train \
  --data_dir /duong_dan/mit_indoor_smartcampus_5 \
  --run_name debug_no_wandb \
  --train_mode head_only \
  --epochs 2 \
  --batch_size 4 \
  --img_size 224 \
  --max_per_class 20
```

## 9. Output sau khi train

Mỗi run tạo thư mục:

```text
outputs/<run_name>/
```

bao gồm:

```text
best_model.pt
history.csv
metrics.json
curves.png
confusion_matrix.png
class_to_idx.json
config.json
```

## 10. W&B

Tên project thống nhất:

```text
csc4005-lab5-mit-indoor-vit
```

Log tối thiểu mỗi epoch:

```text
train_loss
val_loss
train_acc
val_acc
val_macro_f1
lr
epoch_time_sec
```

Log cuối run:

```text
test_acc
test_macro_f1
best_val_acc
best_val_macro_f1
total_params
trainable_params
trainable_ratio
confusion matrix image
learning curves image
```

## 11. Checklist nộp bài

- [ ] Có subset 5 lớp từ MIT Indoor Scenes 67
- [ ] Chạy được ViT baseline ở chế độ `head_only`
- [ ] Có W&B dashboard
- [ ] Có `metrics.json`
- [ ] Có `curves.png`
- [ ] Có `confusion_matrix.png`
- [ ] Có báo cáo theo `REPORT_TEMPLATE.md`
- [ ] Có nhận xét lỗi mô hình và đề xuất cải thiện
