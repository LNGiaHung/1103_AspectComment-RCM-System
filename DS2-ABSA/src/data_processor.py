import json
from typing import List, Dict, Any
import os

class ABSADataProcessor:
    def __init__(self, dataset_name: str, shot_size: int):
        """
        Initialize data processor for ABSA tasks
        Args:
            dataset_name: 'lap' or 'res' or 'res15' or 'res16'
            shot_size: 2 or 5 (for 2% or 5% few-shot)
        """
        self.dataset_name = dataset_name
        self.shot_size = shot_size
        self.data_path = os.path.join('data', dataset_name, f'sample{shot_size}_all.json')
        
    def load_data(self) -> List[Dict[str, Any]]:
        """Load few-shot training data"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        try:
            # Simple approach using utf-8-sig encoding
            with open(self.data_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
                print(f"Successfully loaded {len(data)} examples from {self.data_path}")
                return data
                
        except Exception as e:
            print(f"Error reading file {self.data_path}: {str(e)}")
            raise
            
    def format_for_synthesis(self, examples: List[Dict]) -> List[Dict]:
        """Format examples for synthesis"""
        try:
            formatted = []
            for i, ex in enumerate(examples):
                try:
                    formatted.append({
                        'sentence': ex['sentence'],
                        'aspects': [(asp['target'], asp['polarity']) 
                                  for asp in ex['aspects']]
                    })
                except KeyError as e:
                    print(f"Warning: Missing key {e} in example {i}: {ex}")
                except Exception as e:
                    print(f"Warning: Error processing example {i}: {e}")
            
            print(f"Successfully formatted {len(formatted)} examples")
            return formatted
            
        except Exception as e:
            print(f"Error in format_for_synthesis: {str(e)}")
            raise 