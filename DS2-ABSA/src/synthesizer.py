from typing import List, Dict, Any
import google.generativeai as genai
import json
import os
from tqdm import tqdm

class DualStreamSynthesizer:
    def __init__(self, model_name: str = "gemini-1.5-flash-latest", api_key: str = None):
        """
        Initialize synthesizer with Gemini API key
        Args:
            model_name: Gemini model to use (using the faster flash model like in your notebook)
            api_key: Gemini API key
        """
        self.model_name = model_name
        api_key = api_key if api_key else os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Gemini API key not found")
        
        # Configure Gemini with safety settings like in your notebook
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            self.model_name,
            safety_settings={
                'HATE': 'BLOCK_NONE',
                'HARASSMENT': 'BLOCK_NONE',
                'SEXUAL': 'BLOCK_NONE',
                'DANGEROUS': 'BLOCK_NONE'
            }
        )

    def generate_keypoint_prompt(self, example: Dict) -> str:
        """Generate prompt for key-point driven synthesis"""
        return f"""Given this review example with aspects and sentiments:
Review: {example['sentence']}
Aspects: {example['aspects']}

Generate a new review that maintains similar aspects and sentiments.
You must follow this exact format in your response:
Review: [your generated review]
Aspects: [[aspect1, sentiment1], [aspect2, sentiment2], ...]

Example of good response:
Review: The laptop has excellent battery life but the screen is too dim
Aspects: [["battery life", "positive"], ["screen", "negative"]]"""

    def generate_instance_prompt(self, example: Dict) -> str:
        """Generate prompt for instance-driven synthesis using Vietnamese prompt style like your notebook"""
        return f"""Bạn là một trợ thủ đắc lực! Hãy tạo một đánh giá mới dựa trên đánh giá sau:
Original Review: {example['sentence']}
Original Aspects: {[asp[0] for asp in example['aspects']]}

Yêu cầu:
1. Giữ nguyên các aspects gốc
2. Giữ nguyên cảm xúc cho mỗi aspect
3. Sử dụng từ ngữ và cấu trúc khác

Format phản hồi:
Review: [your generated review]
Aspects: [[aspect1, sentiment1], [aspect2, sentiment2], ...]"""

    async def synthesize(self, example: Dict, stream_type: str) -> Dict:
        """Synthesize new example using Gemini"""
        prompt = (self.generate_keypoint_prompt(example) if stream_type == 'keypoint' 
                 else self.generate_instance_prompt(example))
        
        try:
            # Generate response using Gemini
            response = await self.model.generate_content_async(prompt)
            generated_text = response.text
            
            try:
                # Parse the generated response
                lines = generated_text.split('\n')
                review = next(line.replace('Review:', '').strip() 
                            for line in lines if line.strip().startswith('Review:'))
                aspects_str = next(line.replace('Aspects:', '').strip() 
                                 for line in lines if line.strip().startswith('Aspects:'))
                
                # Clean up and parse aspects
                aspects_str = aspects_str.replace("'", '"')
                aspects = json.loads(aspects_str)
                
                print(f"Successfully generated {stream_type} example")
                return {
                    'sentence': review,
                    'aspects': aspects
                }
                
            except Exception as parse_error:
                print(f"Error parsing Gemini response: {parse_error}")
                print(f"Generated text: {generated_text}")
                return {
                    'sentence': f"Error parsing {stream_type} synthesis response",
                    'aspects': example['aspects']
                }
                
        except Exception as e:
            print(f"Error in {stream_type} synthesis: {str(e)}")
            return {
                'sentence': f"Error in {stream_type} stream synthesis",
                'aspects': example['aspects']
            } 