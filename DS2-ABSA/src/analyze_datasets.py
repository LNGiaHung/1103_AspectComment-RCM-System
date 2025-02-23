from data_processor import ABSADataProcessor
from typing import Dict, List
import json

def analyze_all_datasets():
    datasets = ['lap', 'res', 'res15']
    shot_sizes = [2, 5]
    
    results = {}
    
    for dataset in datasets:
        results[dataset] = {}
        for shot_size in shot_sizes:
            print(f"\nAnalyzing {dataset} with {shot_size}% few-shot")
            
            processor = ABSADataProcessor(dataset, shot_size)
            data = processor.load_all_data()
            stats = processor.get_dataset_stats(data)
            
            results[dataset][f'{shot_size}_shot'] = {
                'statistics': stats,
                'polarity_distribution': get_polarity_distribution(data)
            }
    
    # Save results
    with open('dataset_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nDataset Analysis Summary:")
    print_summary(results)

def get_polarity_distribution(data: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Calculate polarity distribution for each split"""
    dist = {}
    for split, examples in data.items():
        # Initialize with all possible polarities
        polarities = {
            'positive': 0, 
            'negative': 0, 
            'neutral': 0,
            'conflict': 0  # Added conflict polarity
        }
        
        for ex in examples:
            for asp in ex.get('aspects', []):
                pol = asp.get('polarity', 'neutral')
                if pol not in polarities:
                    print(f"Warning: Found unexpected polarity type: {pol}")
                    polarities[pol] = 0
                polarities[pol] += 1
        dist[split] = polarities
    return dist

def print_summary(results: Dict):
    for dataset, shot_results in results.items():
        print(f"\n{dataset.upper()} Dataset:")
        for shot_size, stats in shot_results.items():
            print(f"\n{shot_size}:")
            train_stats = stats['statistics']['train']
            print(f"Train examples: {train_stats['num_examples']}")
            print(f"Average aspects per example: {train_stats['avg_aspects']:.2f}")
            
            dist = stats['polarity_distribution']['train']
            total = sum(dist.values())
            if total > 0:
                print("Polarity distribution:")
                for pol, count in dist.items():
                    if count > 0:  # Only show non-zero polarities
                        print(f"  {pol}: {count/total*100:.1f}%")

if __name__ == "__main__":
    analyze_all_datasets() 