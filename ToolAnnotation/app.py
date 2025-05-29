from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import os
import json
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any
import threading
from functools import wraps
import time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables with type hints
current_data: Optional[pd.DataFrame] = None
current_page: int = 0
rows_per_page: int = 10
modification_cache: Dict[int, Dict[str, Any]] = {}

# File paths
AUTO_SAVE_FILE = 'auto_save_data.json'
BACKUP_DIR = 'backups'
UPLOAD_DIR = 'uploads'
OUTPUT_DIR = 'outputs'

# Thread lock for concurrent access safety
data_lock = threading.Lock()
loading_flag = threading.Event()

# Ensure directories exist
for directory in [BACKUP_DIR, UPLOAD_DIR, OUTPUT_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def error_handler(f):
    """Decorator for consistent error handling"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            return jsonify({'error': f'Internal server error: {str(e)}'}), 500
    return decorated_function

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

def auto_save_json() -> bool:
    """Save current state to JSON file with optimized performance"""
    global current_data, modification_cache
    
    if current_data is None:
        return False
    
    try:
        with data_lock:
            start_time = time.time()
            
            # Create a copy and apply modifications
            temp_data = current_data.copy()
            temp_data = apply_cached_modifications(temp_data)
            
            # Convert to records format efficiently
            output_data = temp_data[['text', 'aspects', 'sentiments']].to_dict(orient='records')
            
            # Create backup with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = os.path.join(BACKUP_DIR, f'backup_{timestamp}.json')
            
            # Save both backup and current auto-save
            for file_path in [backup_file, AUTO_SAVE_FILE]:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Auto-save completed in {time.time() - start_time:.2f}s. Backup: {backup_file}")
            return True
            
    except Exception as e:
        logger.error(f"Auto-save error: {str(e)}")
        return False

def load_from_auto_save() -> bool:
    """Load data from auto-save file if it exists"""
    global current_data, modification_cache, current_page
    
    if not os.path.exists(AUTO_SAVE_FILE):
        return False
        
    try:
        with open(AUTO_SAVE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            return False
            
        # Convert back to DataFrame
        texts = [item.get('text', '') for item in data]
        aspects = [item.get('aspects', []) for item in data]
        sentiments = [item.get('sentiments', {}) for item in data]
        
        current_data = pd.DataFrame({
            'text': texts,
            'aspects': aspects,
            'sentiments': sentiments,
            'is_annotated': [bool(asp or sent) for asp, sent in zip(aspects, sentiments)]
        })
        
        modification_cache.clear()
        current_page = 0
        logger.info(f"Loaded {len(current_data)} rows from auto-save")
        return True
        
    except Exception as e:
        logger.error(f"Error loading auto-save: {str(e)}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@error_handler
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
        except pd.errors.EmptyDataError:
            return jsonify({'error': 'Uploaded CSV file is empty'}), 400
        except pd.errors.ParserError as e:
            return jsonify({'error': f'Could not parse CSV file: {str(e)}'}), 400
        except Exception as e:
            return jsonify({'error': f'Error reading CSV: {str(e)}'}), 400

        # Validate required columns
        if 'text' not in df.columns:
            available_cols = ', '.join(df.columns.tolist())
            return jsonify({'error': f'CSV must contain a "text" column. Available columns: {available_cols}'}), 400

        # Clean and validate data
        df['text'] = df['text'].fillna('').astype(str).str.strip()
        initial_rows = len(df)
        df = df[df['text'].str.len() > 0]
        
        if len(df) == 0:
            return jsonify({'error': 'No valid data rows found after cleaning empty text entries'}), 400
            
        logger.info(f"Cleaned data: {initial_rows} -> {len(df)} rows")

        # Initialize annotation columns
        df['aspects'] = [[] for _ in range(len(df))]
        df['sentiments'] = [{} for _ in range(len(df))]
        df['is_annotated'] = [False for _ in range(len(df))]

        # Reset state
        with data_lock:
            current_data = df
            current_page = 0
            modification_cache.clear()

        # Auto-save
        if not auto_save_json():
            logger.warning("Auto-save failed, but upload completed")

        logger.info(f"Upload completed successfully: {len(current_data)} rows")
        
        return jsonify({
            'message': f'File uploaded successfully! Loaded {len(current_data)} rows.',
            'total_rows': len(current_data),
            'total_pages': (len(current_data) + rows_per_page - 1) // rows_per_page
        })
        
    finally:
        loading_flag.clear()

@app.route('/get_data', methods=['GET'])
@error_handler
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
            'total_rows': len(current_data),
            'cached_modifications': len(modification_cache)
        })

@app.route('/update_data', methods=['POST'])
@error_handler
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
            return jsonify({'error': f'Invalid row index: {row_index}. Max index: {len(current_data) - 1}'}), 400

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

        # Auto-save
        auto_save_json()
        
        logger.info(f"Row {row_index} updated. Cache size: {len(modification_cache)}")
        
        return jsonify({
            'message': 'Data updated successfully',
            'cached_items': len(modification_cache),
            'is_annotated': has_annotations
        })

@app.route('/change_page', methods=['POST'])
@error_handler
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
        return jsonify({'error': f'Invalid page number. Must be between 0 and {total_pages - 1}'}), 400
    
    current_page = new_page
    
    # Auto-save when changing pages
    auto_save_json()
    
    return jsonify({
        'message': f'Changed to page {current_page + 1}',
        'current_page': current_page,
        'total_pages': total_pages
    })

@app.route('/save_data', methods=['POST'])
@error_handler
def save_data():
    """Save final annotated data to CSV files"""
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
            
            # Generate timestamp for unique filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Full CSV with all columns
            full_csv_path = os.path.join(OUTPUT_DIR, f'annotated_data_full_{timestamp}.csv')
            final_data.to_csv(full_csv_path, index=False, encoding='utf-8')
            
            # Simplified CSV with readable format
            simplified_data = pd.DataFrame({
                'text': final_data['text'],
                'aspects': final_data['aspects'].apply(lambda x: ';'.join(x) if x else ''),
                'sentiments': final_data['sentiments'].apply(lambda x: json.dumps(x, ensure_ascii=False) if x else '{}'),
                'is_annotated': final_data['is_annotated']
            })
            
            simplified_csv_path = os.path.join(OUTPUT_DIR, f'annotated_data_simplified_{timestamp}.csv')
            simplified_data.to_csv(simplified_csv_path, index=False, encoding='utf-8')
            
            # Clear modification cache after successful save
            modification_cache.clear()
            
            # Count statistics
            annotated_count = len(final_data[final_data['is_annotated']])
            total_aspects = sum(len(aspects) for aspects in final_data['aspects'] if aspects)
            total_sentiments = sum(len([v for v in sentiments.values() if v]) 
                                 for sentiments in final_data['sentiments'] if sentiments)
            
            logger.info(f"Data saved successfully: {full_csv_path}, {simplified_csv_path}")
            
            return jsonify({
                'message': 'Data saved successfully!',
                'csv_file': f'outputs/{os.path.basename(full_csv_path)}',
                'simplified_csv': f'outputs/{os.path.basename(simplified_csv_path)}',
                'total_rows': len(final_data),
                'annotated_rows': annotated_count,
                'total_aspects': total_aspects,
                'total_sentiments': total_sentiments
            })
            
    except Exception as e:
        logger.error(f"Save error: {str(e)}")
        return jsonify({'error': f'Error saving data: {str(e)}'}), 500

@app.route('/get_stats', methods=['GET'])
@error_handler
def get_stats():
    """Get comprehensive annotation statistics"""
    validation_error = validate_data_loaded()
    if validation_error:
        return validation_error
    
    try:
        with data_lock:
            # Create temporary data with modifications applied
            temp_data = current_data.copy()
            temp_data = apply_cached_modifications(temp_data)
            
            total_rows = len(temp_data)
            annotated_rows = len(temp_data[temp_data['is_annotated']])
            
            # Count aspects and sentiments
            total_aspects = sum(len(aspects) for aspects in temp_data['aspects'] if aspects)
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            total_sentiments = 0
            
            for sentiments in temp_data['sentiments']:
                if isinstance(sentiments, dict):
                    for sentiment in sentiments.values():
                        if sentiment in sentiment_counts:
                            sentiment_counts[sentiment] += 1
                            total_sentiments += 1
            
            completion_percentage = round((annotated_rows / total_rows) * 100, 2) if total_rows > 0 else 0
            
            return jsonify({
                'total_rows': total_rows,
                'annotated_rows': annotated_rows,
                'unannotated_rows': total_rows - annotated_rows,
                'total_aspects': total_aspects,
                'total_sentiments': total_sentiments,
                'sentiment_distribution': sentiment_counts,
                'completion_percentage': completion_percentage,
                'cached_modifications': len(modification_cache)
            })
            
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({'error': f'Error calculating statistics: {str(e)}'}), 500

@app.route('/load_auto_save', methods=['POST'])
@error_handler
def load_auto_save_endpoint():
    """Load data from auto-save file"""
    loading_flag.set()
    try:
        global current_data, modification_cache, current_page
        
        if load_from_auto_save():
            return jsonify({
                'message': 'Auto-save data loaded successfully',
                'total_rows': len(current_data) if current_data is not None else 0
            })
        else:
            return jsonify({'error': 'No auto-save data found or failed to load'}), 404
            
    finally:
        loading_flag.clear()

# File serving for downloads
@app.route('/outputs/<filename>')
def download_file(filename):
    """Serve generated CSV files for download"""
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

if __name__ == '__main__':
    # Try to load auto-save on startup
    if load_from_auto_save():
        logger.info("Auto-save data loaded on startup")
    else:
        logger.info("No auto-save data found on startup")
    
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)