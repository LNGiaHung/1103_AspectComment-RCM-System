# Model Performance Results / Kết Quả Hiệu Suất Mô Hình

## Accuracy Across Datasets / Độ Chính Xác Trên Các Bộ Dữ Liệu

| Dataset | Shot Size | Accuracy | Precision | Recall | F1-Score |
|---------|-----------|----------|-----------|---------|----------|
| LAP | 2% | 47.13% | 69.73% | 47.13% | 54.44% |
| LAP | 5% | 49.79% | 58.89% | 49.79% | 53.22% |
| RES | 2% | 50.45% | 60.36% | 50.45% | 52.44% |
| RES | 5% | 49.90% | 57.86% | 49.90% | 53.41% |
| RES15 | 2% | 42.38% | 48.44% | 42.38% | 44.39% |
| RES15 | 5% | 41.19% | 44.01% | 41.19% | 42.33% |

## Analysis / Phân Tích

### Overall Performance / Hiệu Suất Tổng Thể
- Average F1-Score across all configurations: 50.04%
- Best performing configuration: LAP 2% (F1: 54.44%)

### Notes / Ghi Chú
- Some precision values may be affected by zero division in certain labels
- Results are calculated on aspect-level matching
