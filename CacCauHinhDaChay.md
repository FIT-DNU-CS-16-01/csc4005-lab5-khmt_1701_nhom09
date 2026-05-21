# Các Cấu Hình Đã Chạy – Lab 5: ViT Smart Campus

> Sau khi chạy xong, thêm một dòng vào bảng rồi commit + push để thành viên khác biết.
> Kết quả lấy từ `outputs/<run_name>/metrics.json`. Đặt `run_name` theo MSSV: `<MSSV>_vit_<mô_tả>`.

| # | MSSV | run_name | model_name | train_mode | epochs | batch_size | lr | augment | Ngày chạy | test_acc | test_macro_f1 | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1771040002 | 1771040002_vit_head_only | vit_b_16 | head_only | 10 | 16 | 0.001 | true | 21/05/2026 | 91.38% | 0.8811 | best_val_macro_f1=0.9321 |
| 2 | 1771040002 | 1771040002_vit_finetune | vit_b_16 | finetune | 5 | 8 | 0.00005 | true | 21/05/2026 | 95.69% | 0.9434 | best_epoch=3, overfit epoch5, best_val_macro_f1=0.9434 |
| | | | | | | | | | | | | |
