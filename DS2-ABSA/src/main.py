from data_processor import ABSADataProcessor
import json
import os
import sys
from typing import List, Tuple, Dict

def process_aspects(example: Dict) -> List[Tuple[str, str]]:
    """Process aspects into consistent format"""
    aspects = []
    if 'aspects' in example:
        for asp in example['aspects']:
            if isinstance(asp, dict):
                aspects.append((asp['target'], asp['polarity']))
            elif isinstance(asp, list):
                aspects.append((asp[0], asp[1]))
    return aspects

def process_dataset(dataset: str, shot_size: int):
    """Process a single dataset with given shot size"""
    processor = ABSADataProcessor(dataset, shot_size)
    data = processor.load_all_data()
    
    processed_examples = []
    for example in data['few_shot']:
        if example.get('aspects'):
            processed_examples.append({
                'dataset': dataset,
                'shot_size': shot_size,
                'sentence': example['sentence'],
                'aspects': process_aspects(example)
            })
    return processed_examples

def main():
    try:
        print("\n=== Starting DS2-ABSA Analysis ===")
        
        # 1. Dataset Analysis
        print("1. Analyzing datasets...")
        os.system("python src/analyze_datasets.py")
        
        # 2. Data Processing & Few-shot Sampling
        print("\n2. Processing data...")
        all_processed_data = []
        
        for dataset in ['lap', 'res', 'res15']:
            for shot_size in [2, 5]:
                print(f"\nProcessing {dataset} with {shot_size}% shot")
                processed = process_dataset(dataset, shot_size)
                all_processed_data.extend(processed)
        
        # Save processed data
        output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'processed_data.json')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_processed_data, f, indent=2, ensure_ascii=False)
        
        # 3. Evaluation
        print("\n3. Evaluating results...")
        os.system("python src/evaluate_results.py")
        
        # 4. Generate Final Report
        print("\n4. Generating final report...")
        os.system("python src/combine_results.py")
        
        print("\nAnalysis complete! Check final_results.md for results.")
        
    except Exception as e:
        print("\n=== ERROR OCCURRED ===")
        print(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main() 