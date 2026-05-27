# Các Cấu Hình Đã Chạy – Lab 5: ViT Smart Campus

| # | MSSV | run_name | model_name | train_mode | epochs | batch_size | lr | augment | Ngày chạy | test_acc | test_macro_f1 | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1771040002 | 1771040002_vit_head_only | vit_b_16 | head_only | 10 | 16 | 0.001 | true | 21/05/2026 | 91.38% | 0.8811 | best_val_macro_f1=0.9321 |
| 2 | 1771040002 | 1771040002_vit_finetune | vit_b_16 | finetune | 5 | 8 | 0.00005 | true | 21/05/2026 | 95.69% | 0.9434 | best_epoch=3, overfit epoch5, best_val_macro_f1=0.9434 |
| | | | | | | | | | | | | |
| 3 | 1771040029 | vit_b16_head_only | vit_b_16 | head_only | 50 | 16 | 0.001 | true | 27/05/2026 | 93.97% | 0.9193 | best_epoch=28, best_val_macro_f1=0.9398, lưu dưới tên cũ do lỗi override config trước khi sửa |
| 4 | 1771040029 | 1771040029_vit_finetune_e50 | vit_b_16 | finetune | 50 | 8 | 0.00005 | true | 27/05/2026 | 95.69% | 0.9465 | best_epoch=17, best_val_macro_f1=0.9754, đang là run tốt nhất theo test_macro_f1 |
| 5 | 1771040029 | 1771040029_vit_head_only_noaug_e50 | vit_b_16 | head_only | 50 | 16 | 0.001 | false | 27/05/2026 | 93.10% | 0.9072 | best_epoch=32, best_val_macro_f1=0.9273, kém hơn bản có augmentation |
| 6 | 1771040029 | 1771040029_vit_finetune_lowlr_e50 | vit_b_16 | finetune | 50 | 8 | 0.00002 | true | 27/05/2026 | 95.69% | 0.9433 | best_epoch=15, best_val_macro_f1=0.9757, val tốt nhưng test thấp hơn finetune chuẩn |
| 7 | 1771040029 | 1771040029_resnet18_head_only_e50 | resnet18 | head_only | 50 | 32 | 0.001 | true | 27/05/2026 | 87.93% | 0.8424 | best_epoch=43, best_val_macro_f1=0.9165, thấp hơn ViT rõ rệt |
