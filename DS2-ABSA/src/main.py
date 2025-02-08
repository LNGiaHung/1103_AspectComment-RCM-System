from .data_processor import ABSADataProcessor
from .synthesizer import DualStreamSynthesizer
from .label_refiner import LabelRefiner
from .config import GOOGLE_API_KEY
import asyncio
import json
import os
import sys
import traceback

async def main():
    try:
        print("\n=== Starting ABSA processing ===")
        print(f"Current working directory: {os.getcwd()}")
        
        # Create output directory
        output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created: {output_dir}")
        
        # Initialize components
        print("\nInitializing components...")
        try:
            processor = ABSADataProcessor(dataset_name='lap', shot_size=2)
            print("✓ Data processor initialized")
        except Exception as e:
            print(f"Error initializing data processor: {str(e)}")
            raise
            
        try:
            print(f"Gemini API Key (first 10 chars): {GOOGLE_API_KEY[:10]}...")
            synthesizer = DualStreamSynthesizer(api_key=GOOGLE_API_KEY)
            print("✓ Synthesizer initialized")
        except Exception as e:
            print(f"Error initializing synthesizer: {str(e)}")
            raise
            
        try:
            refiner = LabelRefiner()
            print("✓ Label refiner initialized")
        except Exception as e:
            print(f"Error initializing label refiner: {str(e)}")
            raise
        
        # Load and process data
        print("\nLoading data...")
        try:
            raw_data = processor.load_data()
            print(f"✓ Loaded {len(raw_data)} examples")
            print(f"Data path: {processor.data_path}")
            
            if not raw_data:
                raise ValueError("No data loaded")
                
            formatted_data = processor.format_for_synthesis(raw_data)
            print(f"✓ Formatted {len(formatted_data)} examples for synthesis")
            
            # Print first example as sanity check
            print("\nFirst example:")
            print(json.dumps(formatted_data[0], indent=2))
            
        except FileNotFoundError:
            print(f"ERROR: Data file not found at {processor.data_path}")
            raise
        except Exception as e:
            print(f"Error loading/formatting data: {str(e)}")
            raise
        
        # Synthesize new examples
        print("\nStarting synthesis...")
        synthetic_data = []
        for i, example in enumerate(formatted_data, 1):
            print(f"Processing example {i}/{len(formatted_data)}")
            print(f"  - Generating keypoint-driven example...")
            keypoint_ex = await synthesizer.synthesize(example, 'keypoint')
            print(f"  - Generating instance-driven example...")
            instance_ex = await synthesizer.synthesize(example, 'instance')
            synthetic_data.extend([keypoint_ex, instance_ex])
        
        print(f"\nGenerated {len(synthetic_data)} synthetic examples")
        
        # Refine labels
        print("\nNormalizing labels...")
        normalized_data = refiner.normalize_labels(synthetic_data)
        print("Applying self-training...")
        refined_data = refiner.self_train(normalized_data)
        
        # Save results
        output_file = os.path.join(output_dir, 'synthetic_refined_data.json')
        print(f"\nSaving results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(refined_data, f, indent=2, ensure_ascii=False)
        
        print("\n=== Processing complete! ===")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print("\n=== ERROR OCCURRED ===")
        print("Error details:")
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1) 