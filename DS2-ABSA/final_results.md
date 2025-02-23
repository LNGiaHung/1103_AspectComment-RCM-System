# ABSA System Final Results / Kết Quả Cuối Cùng Hệ Thống ABSA

## 1. Dataset Statistics / Thống Kê Dữ Liệu

### Laptop Reviews (LAP)
- Training: 2,436 examples
- Test: 800 examples
- Few-shot samples:
  - 2%: 50 examples
  - 5%: 122 examples
- Average aspects per example: 0.79

### Restaurant Reviews (RES)
- Training: 2,432 examples
- Test: 800 examples
- Few-shot samples:
  - 2%: 49 examples
  - 5%: 122 examples
- Average aspects per example: 1.22

### Restaurant Reviews 2015 (RES15)
- Training: 1,052 examples
- Test: 685 examples
- Few-shot samples:
  - 2%: 21 examples
  - 5%: 53 examples
- Average aspects per example: 0.91

## 2. Evaluation Results / Kết Quả Đánh Giá

| Dataset | Shot Size | Accuracy | Precision | Recall | F1-Score |
|---------|-----------|----------|-----------|---------|----------|
| LAP | 2% | 47.13% | 69.73% | 47.13% | 54.44% |
| LAP | 5% | 49.79% | 58.89% | 49.79% | 53.22% |
| RES | 2% | 50.45% | 60.36% | 50.45% | 52.44% |
| RES | 5% | 49.90% | 57.86% | 49.90% | 53.41% |
| RES15 | 2% | 42.38% | 48.44% | 42.38% | 44.39% |
| RES15 | 5% | 41.19% | 44.01% | 41.19% | 42.33% |

## 3. Analysis / Phân Tích

### Key Findings / Phát Hiện Chính
1. Dataset Balance / Cân Bằng Dữ Liệu:
   - LAP: Most balanced distribution
   - RES15: Highly skewed toward positive

2. Aspect Density / Mật Độ Khía Cạnh:
   - RES: Highest (1.22 aspects/example)
   - LAP: Lowest (0.79 aspects/example)

3. Performance Impact / Ảnh Hưởng Hiệu Suất:
   - 5% samples generally provide better results
   - RES15 2% (21 samples) may be too small for reliable evaluation

### Overall Performance / Hiệu Suất Tổng Thể
- Average F1-Score: 50.04%
- Best performing dataset: LAP (F1: 54.44%)
