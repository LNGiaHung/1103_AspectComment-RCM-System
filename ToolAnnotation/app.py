from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import os
import json
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any
import threading

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
current_data: Optional[pd.DataFrame] = None
current_page: int = 0
rows_per_page: int = 10
modification_cache: Dict[int, Dict[str, Any]] = {}

# File paths
OUTPUT_DIR = 'outputs'

# Thread lock for concurrent access safety
data_lock = threading.Lock()
loading_flag = threading.Event()

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def validate_data_loaded():
    """Check if data is loaded"""
    if current_data is None:
        return jsonify({'error': 'No data loaded. Please upload a CSV file first.'}), 400
    return None

def apply_cached_modifications(data_copy: pd.DataFrame) -> pd.DataFrame:
    """Apply all cached modifications to a DataFrame copy"""
    for row_idx, modifications in modification_cache.items():
        if row_idx < len(data_copy):
            if 'aspects' in modifications:
                data_copy.at[row_idx, 'aspects'] = modifications['aspects']
            if 'sentiments' in modifications:
                data_copy.at[row_idx, 'sentiments'] = modifications['sentiments']
            if 'is_annotated' in modifications:
                data_copy.at[row_idx, 'is_annotated'] = modifications['is_annotated']
    return data_copy

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    loading_flag.set()
    try:
        global current_data, current_page, modification_cache
        
        logger.info('Upload started')
        
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400
            
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400

        # Read and validate CSV
        try:
            df = pd.read_csv(file)
            logger.info(f"CSV loaded successfully. Shape: {df.shape}")
        except Exception as e:
            return jsonify({'error': f'Error reading CSV: {str(e)}'}), 400

        # Validate required columns
        if 'text' not in df.columns:
            return jsonify({'error': 'CSV must contain a "text" column'}), 400

        # Clean and validate data
        df['text'] = df['text'].fillna('').astype(str).str.strip()
        df = df[df['text'].str.len() > 0]
        
        if len(df) == 0:
            return jsonify({'error': 'No valid data rows found'}), 400

        # Initialize annotation columns
        df['aspects'] = [[] for _ in range(len(df))]
        df['sentiments'] = [{} for _ in range(len(df))]
        df['is_annotated'] = [False for _ in range(len(df))]

        # Reset state
        with data_lock:
            current_data = df
            current_page = 0
            modification_cache.clear()

        logger.info(f"Upload completed successfully: {len(current_data)} rows")
        
        return jsonify({
            'message': f'File uploaded successfully! Loaded {len(current_data)} rows.',
            'total_rows': len(current_data),
            'total_pages': (len(current_data) + rows_per_page - 1) // rows_per_page
        })
        
    finally:
        loading_flag.clear()

@app.route('/get_data', methods=['GET'])
def get_data():
    """Get paginated data with current annotations"""
    validation_error = validate_data_loaded()
    if validation_error:
        return validation_error
    
    with data_lock:
        start_idx = current_page * rows_per_page
        end_idx = min(start_idx + rows_per_page, len(current_data))
        
        # Get page data and apply cached modifications
        page_data = current_data.iloc[start_idx:end_idx].copy()
        page_data = apply_cached_modifications(page_data)
        
        # Format response data
        result_data = []
        for idx, (_, row) in enumerate(page_data.iterrows()):
            row_dict = {
                'text': str(row['text']),
                'aspects': row.get('aspects', []),
                'sentiments': row.get('sentiments', {}),
                'is_annotated': row.get('is_annotated', False),
                'row_index': start_idx + idx
            }
            result_data.append(row_dict)
        
        total_pages = (len(current_data) + rows_per_page - 1) // rows_per_page
        
        return jsonify({
            'data': result_data,
            'current_page': current_page,
            'total_pages': total_pages,
            'has_prev': current_page > 0,
            'has_next': current_page < total_pages - 1,
            'start_idx': start_idx + 1,
            'end_idx': end_idx,
            'total_rows': len(current_data)
        })

@app.route('/update_data', methods=['POST'])
def update_data():
    """Update annotations for a specific row"""
    if loading_flag.is_set():
        return jsonify({'error': 'Data is currently loading. Please try again later.'}), 423
    
    validation_error = validate_data_loaded()
    if validation_error:
        return validation_error
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
        
    row_index = data.get('rowIndex')
    modifications = data.get('modifications', {})
    
    if row_index is None:
        return jsonify({'error': 'Row index is required'}), 400
        
    if not isinstance(row_index, int) or row_index < 0:
        return jsonify({'error': 'Row index must be a non-negative integer'}), 400

    with data_lock:
        if row_index >= len(current_data):
            return jsonify({'error': f'Invalid row index: {row_index}'}), 400

        # Initialize cache entry if not exists
        if row_index not in modification_cache:
            modification_cache[row_index] = {
                'aspects': list(current_data.at[row_index, 'aspects']),
                'sentiments': dict(current_data.at[row_index, 'sentiments']),
                'is_annotated': current_data.at[row_index, 'is_annotated']
            }

        # Apply modifications
        if 'aspects' in modifications:
            modification_cache[row_index]['aspects'] = modifications['aspects']
            
        if 'sentiments' in modifications:
            modification_cache[row_index]['sentiments'].update(modifications['sentiments'])
            
        # Mark as annotated if has aspects or sentiments
        has_annotations = (
            bool(modification_cache[row_index]['aspects']) or 
            bool([v for v in modification_cache[row_index]['sentiments'].values() if v])
        )
        modification_cache[row_index]['is_annotated'] = has_annotations
        
        return jsonify({
            'message': 'Data updated successfully',
            'is_annotated': has_annotations
        })

@app.route('/change_page', methods=['POST'])
def change_page():
    """Change current page"""
    global current_page
    
    validation_error = validate_data_loaded()
    if validation_error:
        return validation_error
    
    data = request.get_json()
    if not data or 'page' not in data:
        return jsonify({'error': 'Page parameter is required'}), 400
    
    new_page = data.get('page')
    if not isinstance(new_page, int):
        return jsonify({'error': 'Page must be an integer'}), 400
    
    total_pages = (len(current_data) + rows_per_page - 1) // rows_per_page
    if new_page < 0 or new_page >= total_pages:
        return jsonify({'error': f'Invalid page number'}), 400
    
    current_page = new_page
    return jsonify({
        'message': f'Changed to page {current_page + 1}',
        'current_page': current_page,
        'total_pages': total_pages
    })

@app.route('/export_csv', methods=['POST'])
def export_csv():
    """Export annotated data as CSV file"""
    if loading_flag.is_set():
        return jsonify({'error': 'Data is currently loading. Please try again later.'}), 423
    
    validation_error = validate_data_loaded()
    if validation_error:
        return validation_error

    try:
        with data_lock:
            # Apply all cached modifications to final data
            final_data = current_data.copy()
            final_data = apply_cached_modifications(final_data)
            
            # Generate timestamp for unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save as CSV with readable format
            simplified_data = pd.DataFrame({
                'text': final_data['text'],
                'aspects': final_data['aspects'].apply(lambda x: ';'.join(x) if x else ''),
                'sentiments': final_data['sentiments'].apply(lambda x: json.dumps(x, ensure_ascii=False) if x else '{}'),
                'is_annotated': final_data['is_annotated']
            })
            
            csv_path = os.path.join(OUTPUT_DIR, f'annotated_data_{timestamp}.csv')
            simplified_data.to_csv(csv_path, index=False, encoding='utf-8')
            
            # Count statistics
            annotated_count = len(final_data[final_data['is_annotated']])
            
            return jsonify({
                'message': 'Data exported to CSV successfully!',
                'csv_file': f'outputs/{os.path.basename(csv_path)}',
                'total_rows': len(final_data),
                'annotated_rows': annotated_count
            })
            
    except Exception as e:
        logger.error(f"CSV export error: {str(e)}")
        return jsonify({'error': f'Error exporting CSV: {str(e)}'}), 500

@app.route('/outputs/<filename>')
def download_file(filename):
    """Serve generated files for download"""
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)