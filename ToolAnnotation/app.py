from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import json
from datetime import datetime

app = Flask(__name__)

# Global variables
current_data = None
current_page = 0
rows_per_page = 10
# Cache to store modifications
modification_cache = {}
# Auto-save file path
AUTO_SAVE_FILE = 'auto_save_data.json'
BACKUP_DIR = 'backups'

# Ensure backup directory exists
if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)

def auto_save():
    """Save current state to auto-save file"""
    if current_data is None:
        return
    
    try:
        # Apply all cached modifications
        temp_data = current_data.copy()
        for row_idx, modifications in modification_cache.items():
            temp_data.at[row_idx, 'aspects'] = modifications['aspects']
            temp_data.at[row_idx, 'sentiments'] = modifications['sentiments']
        
        # Save current state
        output_data = []
        for _, row in temp_data.iterrows():
            entry = {
                'text': row['text'],
                'aspects': row['aspects'],
                'sentiments': row['sentiments']
            }
            output_data.append(entry)
        
        # Save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = os.path.join(BACKUP_DIR, f'backup_{timestamp}.json')
        
        # Save backup
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # Save current state
        with open(AUTO_SAVE_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Auto-save error: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_data, current_page
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            current_data = pd.read_csv(file)
            # Ensure required columns exist
            if 'text' not in current_data.columns:
                return jsonify({'error': 'CSV must contain a "text" column'}), 400
            
            # Initialize aspect and sentiment columns if they don't exist
            if 'aspects' not in current_data.columns:
                current_data['aspects'] = current_data.apply(lambda x: [], axis=1)
            if 'sentiments' not in current_data.columns:
                current_data['sentiments'] = current_data.apply(lambda x: [], axis=1)
            
            current_page = 0
            total_rows = len(current_data)
            return jsonify({
                'message': f'Successfully loaded {total_rows} rows',
                'total_rows': total_rows
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/get_data', methods=['GET'])
def get_data():
    global current_data, current_page, modification_cache
    if current_data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    start_idx = current_page * rows_per_page
    end_idx = min(start_idx + rows_per_page, len(current_data))
    
    # Get the page data and apply any cached modifications
    page_data = current_data.iloc[start_idx:end_idx].to_dict('records')
    for idx, row in enumerate(page_data):
        row_idx = start_idx + idx
        if row_idx in modification_cache:
            row['aspects'] = modification_cache[row_idx]['aspects']
            row['sentiments'] = modification_cache[row_idx]['sentiments']
    
    total_pages = (len(current_data) + rows_per_page - 1) // rows_per_page
    
    # Get navigation info
    has_prev = current_page > 0
    has_next = current_page < total_pages - 1
    
    return jsonify({
        'data': page_data,
        'current_page': current_page,
        'total_pages': total_pages,
        'has_prev': has_prev,
        'has_next': has_next,
        'start_idx': start_idx + 1,  # 1-based index for display
        'end_idx': end_idx,
        'total_rows': len(current_data)
    })

@app.route('/update_data', methods=['POST'])
def update_data():
    global current_data, modification_cache
    if current_data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    data = request.json
    row_index = data.get('row_index')
    aspects = data.get('aspects', [])
    sentiments = data.get('sentiments', [])
    
    if row_index is None:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        # Store the modification in cache
        modification_cache[row_index] = {
            'aspects': aspects,
            'sentiments': sentiments
        }
        
        # Auto-save after each modification
        auto_save()
        
        return jsonify({
            'message': 'Data cached and auto-saved successfully',
            'row_index': row_index,
            'aspects': aspects,
            'sentiments': sentiments
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/change_page', methods=['POST'])
def change_page():
    global current_page, current_data, modification_cache
    if current_data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    data = request.json
    new_page = data.get('page')
    
    if new_page is None:
        return jsonify({'error': 'Missing page parameter'}), 400
    
    # Validate page number
    total_pages = (len(current_data) + rows_per_page - 1) // rows_per_page
    if new_page < 0 or new_page >= total_pages:
        return jsonify({'error': 'Invalid page number'}), 400
    
    # Apply cached modifications to the current data before changing pages
    for row_idx, modifications in modification_cache.items():
        current_data.at[row_idx, 'aspects'] = modifications['aspects']
        current_data.at[row_idx, 'sentiments'] = modifications['sentiments']
    
    current_page = new_page
    
    # Auto-save when changing pages
    auto_save()
    
    return jsonify({
        'message': f'Changed to page {current_page}',
        'current_page': current_page,
        'total_pages': total_pages
    })

@app.route('/get_page_info', methods=['GET'])
def get_page_info():
    """Get information about current page and navigation"""
    global current_data, current_page
    if current_data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    total_pages = (len(current_data) + rows_per_page - 1) // rows_per_page
    start_idx = current_page * rows_per_page
    end_idx = min(start_idx + rows_per_page, len(current_data))
    
    return jsonify({
        'current_page': current_page,
        'total_pages': total_pages,
        'has_prev': current_page > 0,
        'has_next': current_page < total_pages - 1,
        'start_idx': start_idx + 1,  # 1-based index for display
        'end_idx': end_idx,
        'total_rows': len(current_data)
    })

@app.route('/save_data', methods=['POST'])
def save_data():
    global current_data, modification_cache
    if current_data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    try:
        # Apply all cached modifications before saving
        for row_idx, modifications in modification_cache.items():
            current_data.at[row_idx, 'aspects'] = modifications['aspects']
            current_data.at[row_idx, 'sentiments'] = modifications['sentiments']
        
        # Save in ABSA format
        output_data = []
        for _, row in current_data.iterrows():
            entry = {
                'text': row['text'],
                'aspects': row['aspects'],
                'sentiments': row['sentiments']
            }
            output_data.append(entry)
        
        # Save as JSON with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'annotated_data_{timestamp}.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # Clear the cache after saving
        modification_cache.clear()
        
        return jsonify({
            'message': 'Data saved successfully',
            'file': output_file
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 