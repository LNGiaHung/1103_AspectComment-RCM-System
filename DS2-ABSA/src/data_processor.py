import json
from typing import List, Dict, Any
import os

class ABSADataProcessor:
    def __init__(self, dataset_name: str, shot_size: int):
        """
        Initialize data processor for ABSA tasks
        Args:
            dataset_name: 'lap' or 'res' or 'res15'
            shot_size: 2 or 5 (for 2% or 5% few-shot)
        """
        self.dataset_name = dataset_name
        self.shot_size = shot_size
        self.data_path = os.path.join('data', dataset_name, f'sample{shot_size}_all.json')
        self.train_path = os.path.join('data', dataset_name, 'train_all.json')
        self.test_path = os.path.join('data', dataset_name, 'test_all.json')
        
    def load_data(self) -> List[Dict[str, Any]]:
        """Load few-shot training data (maintained for backwards compatibility)"""
        return self.load_file(self.data_path)
        
    def load_all_data(self) -> Dict[str, List[Dict]]:
        """Load all data splits"""
        data = {
            'train': self.load_file(self.train_path),
            'test': self.load_file(self.test_path),
            'few_shot': self.load_file(self.data_path)
        }
        return data
        
    def load_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from a specific file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
                print(f"Successfully loaded {len(data)} examples from {file_path}")
                return data
                
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            raise

    def get_dataset_stats(self, data: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """Get statistics for each dataset split"""
        stats = {}
        for split, examples in data.items():
            total_aspects = sum(len(ex.get('aspects', [])) for ex in examples)
            stats[split] = {
                'num_examples': len(examples),
                'num_aspects': total_aspects,
                'avg_aspects': total_aspects / len(examples) if examples else 0
            }
        return stats

    def format_for_synthesis(self, examples: List[Dict]) -> List[Dict]:
        """Format examples for synthesis"""
        try:
            formatted = []
            for i, ex in enumerate(examples):
                try:
                    formatted.append({
                        'sentence': ex['sentence'],
                        'aspects': [(asp['target'], asp['polarity']) 
                                  for asp in ex.get('aspects', [])]
                    })
                except KeyError as e:
                    print(f"Warning: Missing key {e} in example {i}")
                except Exception as e:
                    print(f"Warning: Error processing example {i}: {e}")
            
            print(f"Successfully formatted {len(formatted)} examples")
            return formatted
            
        except Exception as e:
            print(f"Error in format_for_synthesis: {str(e)}")
            raise 