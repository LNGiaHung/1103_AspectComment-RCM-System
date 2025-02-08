import google.generativeai as genai
import os
from google.api_core import retry

# Configure the API
api_key = 'YOUR_NEW_API_KEY_HERE'  # Paste your new API key from Google AI Studio
genai.configure(api_key=api_key)

def test_gemini():
    try:
        # Test a simple generation
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content('Say "Hello, World!"')
        print("Response:", response.text)
        print("\nAPI test successful!")
        
    except Exception as e:
        print(f"\nError testing Gemini API: {str(e)}")
        print("\nPlease make sure:")
        print("1. You've enabled the Gemini API")
        print("2. Your API key is correct")
        print("3. You have billing set up (if required)")

if __name__ == "__main__":
    test_gemini() 