import json
from typing import Dict, List
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_results() -> Dict:
    """Load processed results and test data"""
    results = {}
    
    # Load processed results
    proc_path = os.path.join('output', 'processed_data.json')
    try:
        with open(proc_path, 'r', encoding='utf-8') as f:
            all_predictions = json.load(f)
            # Group predictions by dataset
            results['predictions'] = {
                'lap': [p for p in all_predictions if p['dataset'] == 'lap'],
                'res': [p for p in all_predictions if p['dataset'] == 'res'],
                'res15': [p for p in all_predictions if p['dataset'] == 'res15']
            }
            print(f"Loaded predictions from {proc_path}")
    except FileNotFoundError:
        print(f"No processed data found at {proc_path}")
        return None
        
    # Load test data for each dataset
    datasets = ['lap', 'res', 'res15']
    for dataset in datasets:
        test_path = os.path.join('data', dataset, 'test_all.json')
        try:
            with open(test_path, 'r', encoding='utf-8-sig') as f:
                results[f'{dataset}_test'] = json.load(f)
                print(f"Loaded test data from {test_path}")
        except FileNotFoundError:
            print(f"No test data found for {dataset}")
            
    return results

def group_predictions_by_dataset(predictions: List[Dict]) -> Dict[str, List[Dict]]:
    """Group predictions by dataset based on content patterns"""
    dataset_predictions = {
        'lap': [],
        'res': [],
        'res15': []
    }
    
    for pred in predictions:
        # Check for laptop-related keywords
        if any(keyword in pred['sentence'].lower() for keyword in 
               ['laptop', 'computer', 'pc', 'screen', 'keyboard', 'battery']):
            dataset_predictions['lap'].append(pred)
        
        # Check for restaurant-related keywords
        elif any(keyword in pred['sentence'].lower() for keyword in 
                ['restaurant', 'food', 'menu', 'service', 'dish', 'waiter']):
            # Check if it's likely res15 based on ID pattern or specific features
            if 'ID' in pred and ':' in str(pred['ID']):
                dataset_predictions['res15'].append(pred)
            else:
                dataset_predictions['res'].append(pred)
                
        # Default to the dataset with most similar patterns
        else:
            # Add to the dataset that has most similar aspect patterns
            dataset_predictions['lap'].append(pred)  # Default to lap if uncertain
    
    return dataset_predictions

def evaluate_aspects(true_data: List[Dict], pred_data: List[Dict]) -> Dict:
    """Calculate aspect-level metrics"""
    y_true = []
    y_pred = []
    
    # Match predictions with test data based on aspects
    for true_ex in true_data:
        true_aspects = [(asp['target'], asp['polarity']) 
                       for asp in true_ex.get('aspects', [])]
        if not true_aspects:
            continue
            
        # Find matching prediction by aspect overlap
        for pred in pred_data:
            pred_aspects = pred.get('aspects', [])
            if not pred_aspects:
                continue
                
            # Convert aspects to same format if needed
            if isinstance(pred_aspects[0], list):
                pred_aspects = [(asp[0], asp[1]) for asp in pred_aspects]
            else:
                pred_aspects = [(asp['target'], asp['polarity']) 
                              for asp in pred_aspects]
            
            # Check for aspect overlap
            for true_asp in true_aspects:
                for pred_asp in pred_aspects:
                    if similar_aspects(true_asp[0], pred_asp[0]):
                        y_true.append(true_asp[1])
                        y_pred.append(pred_asp[1])
    
    if not y_true or not y_pred:
        print("Warning: No matching aspects found for evaluation")
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0
        }
    
    # Add zero_division parameter to handle the warning
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, 
        average='weighted',
        zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    
    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }

def similar_aspects(aspect1: str, aspect2: str, threshold: float = 0.8) -> bool:
    """Check if two aspects are similar using token overlap"""
    # Normalize aspects
    a1 = set(aspect1.lower().strip().split())
    a2 = set(aspect2.lower().strip().split())
    
    # Calculate Jaccard similarity
    intersection = len(a1.intersection(a2))
    union = len(a1.union(a2))
    
    return intersection / union >= threshold if union > 0 else False

def main():
    # Load all results
    results = load_results()
    if not results:
        print("No results to evaluate")
        return
        
    metrics = {}
    predictions = results['predictions']
    
    # Evaluate for each dataset
    for dataset in ['lap', 'res', 'res15']:
        if f'{dataset}_test' in results and dataset in predictions:
            test_data = results[f'{dataset}_test']
            dataset_preds = predictions[dataset]
            
            if dataset_preds:
                metrics[dataset] = {}
                
                # Evaluate 2% and 5% separately
                for shot_size in [2, 5]:
                    shot_preds = [p for p in dataset_preds if p['shot_size'] == shot_size]
                    
                    print(f"\nEvaluating {dataset.upper()} {shot_size}%:")
                    print(f"Test examples: {len(test_data)}")
                    print(f"Predictions: {len(shot_preds)}")
                    print(f"Test aspects: {sum(len(ex.get('aspects', [])) for ex in test_data)}")
                    print(f"Predicted aspects: {sum(len(ex.get('aspects', [])) for ex in shot_preds)}")
                    
                    shot_metrics = evaluate_aspects(test_data, shot_preds)
                    metrics[dataset][f'{shot_size}%'] = shot_metrics
                    
                    print(f"Results for {shot_size}%:")
                    print(f"Accuracy: {shot_metrics['accuracy']:.2f}%")
                    print(f"F1-Score: {shot_metrics['f1']:.2f}%")
                    print(f"Precision: {shot_metrics['precision']:.2f}%")
                    print(f"Recall: {shot_metrics['recall']:.2f}%")
            else:
                print(f"\nNo predictions found for {dataset}")
    
    # Save detailed results
    with open('evaluation_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate markdown report
    generate_markdown_report(metrics)

def generate_markdown_report(results: Dict):
    with open('model_performance.md', 'w', encoding='utf-8') as f:
        f.write("# Model Performance Results / Kết Quả Hiệu Suất Mô Hình\n\n")
        
        f.write("## Accuracy Across Datasets / Độ Chính Xác Trên Các Bộ Dữ Liệu\n\n")
        
        # Create results table
        f.write("| Dataset | Shot Size | Accuracy | Precision | Recall | F1-Score |\n")
        f.write("|---------|-----------|----------|-----------|---------|----------|\n")
        
        for dataset in ['lap', 'res', 'res15']:
            if dataset in results:
                # Write 2% results
                metrics_2 = results[dataset].get('2%', {})
                f.write(f"| {dataset.upper()} | 2% | ")
                f.write(f"{metrics_2.get('accuracy', 0):.2f}% | ")
                f.write(f"{metrics_2.get('precision', 0):.2f}% | ")
                f.write(f"{metrics_2.get('recall', 0):.2f}% | ")
                f.write(f"{metrics_2.get('f1', 0):.2f}% |\n")
                
                # Write 5% results
                metrics_5 = results[dataset].get('5%', {})
                f.write(f"| {dataset.upper()} | 5% | ")
                f.write(f"{metrics_5.get('accuracy', 0):.2f}% | ")
                f.write(f"{metrics_5.get('precision', 0):.2f}% | ")
                f.write(f"{metrics_5.get('recall', 0):.2f}% | ")
                f.write(f"{metrics_5.get('f1', 0):.2f}% |\n")
        
        f.write("\n## Analysis / Phân Tích\n\n")
        
        # Add overall analysis
        f.write("### Overall Performance / Hiệu Suất Tổng Thể\n")
        if results:
            # Calculate average F1 across all datasets and shot sizes
            total_f1 = 0
            count = 0
            best_f1 = 0
            best_config = ''
            
            for dataset, shot_metrics in results.items():
                for shot_size, metrics in shot_metrics.items():
                    f1 = metrics.get('f1', 0)
                    total_f1 += f1
                    count += 1
                    if f1 > best_f1:
                        best_f1 = f1
                        best_config = f"{dataset.upper()} {shot_size}"
            
            avg_f1 = total_f1 / count if count > 0 else 0
            f.write(f"- Average F1-Score across all configurations: {avg_f1:.2f}%\n")
            f.write(f"- Best performing configuration: {best_config} (F1: {best_f1:.2f}%)\n")
            
            # Add warning about zero division if needed
            f.write("\n### Notes / Ghi Chú\n")
            f.write("- Some precision values may be affected by zero division in certain labels\n")
            f.write("- Results are calculated on aspect-level matching\n")
        else:
            f.write("No results available for analysis\n")

if __name__ == "__main__":
    main() 