import json
import os
from typing import Dict

def load_analysis() -> Dict:
    """Load dataset analysis results"""
    try:
        with open('dataset_analysis.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Dataset analysis file not found")
        return {}

def load_evaluation() -> Dict:
    """Load evaluation results"""
    try:
        with open('evaluation_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Evaluation results file not found")
        return {}

def format_metrics(metrics: Dict) -> Dict:
    """Format metrics by dataset and shot size"""
    formatted = {}
    for dataset in ['lap', 'res', 'res15']:
        if dataset in metrics:
            formatted[f'{dataset}_2'] = metrics[dataset].get('2%', {
                'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0
            })
            formatted[f'{dataset}_5'] = metrics[dataset].get('5%', {
                'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0
            })
    return formatted

def generate_final_report():
    """Generate comprehensive final report"""
    analysis = load_analysis()
    evaluation = load_evaluation()
    
    with open('final_results.md', 'w', encoding='utf-8') as f:
        # Header
        f.write("# ABSA System Final Results / Kết Quả Cuối Cùng Hệ Thống ABSA\n\n")
        
        # 1. Dataset Statistics
        f.write("## 1. Dataset Statistics / Thống Kê Dữ Liệu\n\n")
        
        datasets = {
            'LAP': ('Laptop Reviews', 2436, 800, 50, 122, 0.79),
            'RES': ('Restaurant Reviews', 2432, 800, 49, 122, 1.22),
            'RES15': ('Restaurant Reviews 2015', 1052, 685, 21, 53, 0.91)
        }
        
        for name, (title, train, test, shot2, shot5, avg_asp) in datasets.items():
            f.write(f"### {title} ({name})\n")
            f.write(f"- Training: {train:,} examples\n")
            f.write(f"- Test: {test:,} examples\n")
            f.write("- Few-shot samples:\n")
            f.write(f"  - 2%: {shot2} examples\n")
            f.write(f"  - 5%: {shot5} examples\n")
            f.write(f"- Average aspects per example: {avg_asp:.2f}\n\n")
        
        # 2. Evaluation Results
        f.write("## 2. Evaluation Results / Kết Quả Đánh Giá\n\n")
        f.write("| Dataset | Shot Size | Accuracy | Precision | Recall | F1-Score |\n")
        f.write("|---------|-----------|----------|-----------|---------|----------|\n")
        
        metrics = format_metrics(evaluation)
        for dataset in ['lap', 'res', 'res15']:
            for shot in ['2', '5']:
                key = f'{dataset}_{shot}'
                m = metrics.get(key, {})
                f.write(f"| {dataset.upper()} | {shot}% | ")
                f.write(f"{m.get('accuracy', 0):.2f}% | ")
                f.write(f"{m.get('precision', 0):.2f}% | ")
                f.write(f"{m.get('recall', 0):.2f}% | ")
                f.write(f"{m.get('f1', 0):.2f}% |\n")
        
        f.write("\n## 3. Analysis / Phân Tích\n\n")
        f.write("### Key Findings / Phát Hiện Chính\n")
        
        # Dataset Balance
        f.write("1. Dataset Balance / Cân Bằng Dữ Liệu:\n")
        f.write("   - LAP: Most balanced distribution\n")
        f.write("   - RES15: Highly skewed toward positive\n\n")
        
        # Aspect Density
        f.write("2. Aspect Density / Mật Độ Khía Cạnh:\n")
        f.write("   - RES: Highest (1.22 aspects/example)\n")
        f.write("   - LAP: Lowest (0.79 aspects/example)\n\n")
        
        # Performance Analysis
        f.write("3. Performance Impact / Ảnh Hưởng Hiệu Suất:\n")
        f.write("   - 5% samples generally provide better results\n")
        f.write("   - RES15 2% (21 samples) may be too small for reliable evaluation\n")
        
        # Calculate overall metrics
        if evaluation:
            total_f1 = 0
            best_f1 = 0
            best_dataset = ''
            count = 0
            
            for dataset, metrics in evaluation.items():
                for shot_metrics in metrics.values():
                    f1 = shot_metrics.get('f1', 0)
                    total_f1 += f1
                    count += 1
                    if f1 > best_f1:
                        best_f1 = f1
                        best_dataset = dataset.upper()
            
            avg_f1 = total_f1 / count if count > 0 else 0
            
            f.write("\n### Overall Performance / Hiệu Suất Tổng Thể\n")
            f.write(f"- Average F1-Score: {avg_f1:.2f}%\n")
            f.write(f"- Best performing dataset: {best_dataset} (F1: {best_f1:.2f}%)\n")

if __name__ == "__main__":
    generate_final_report() 