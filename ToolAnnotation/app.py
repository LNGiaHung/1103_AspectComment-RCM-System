from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import pandas as pd
import os
import tempfile
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# Store labeled data (in a real application, this would be a database)
labeled_data = []
current_texts = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "File must be a CSV"}), 400

    try:
        # Read CSV file
        df = pd.read_csv(file)
        # Assuming the text column is named 'text', adjust if different
        if 'text' not in df.columns:
            return jsonify({"error": "CSV must contain a 'text' column"}), 400
        
        # Store texts for labeling
        global current_texts
        current_texts = df['text'].tolist()
        
        return jsonify({
            "status": "success",
            "message": f"Successfully loaded {len(current_texts)} texts",
            "texts": current_texts
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/get_texts', methods=['GET'])
def get_texts():
    return jsonify({"texts": current_texts})

@app.route('/api/save_label', methods=['POST'])
def save_label():
    data = request.json
    labeled_data.append(data)
    return jsonify({"status": "success", "message": "Label saved successfully"})

@app.route('/api/get_labels', methods=['GET'])
def get_labels():
    return jsonify(labeled_data)

@app.route('/api/export_csv', methods=['GET'])
def export_csv():
    if not labeled_data:
        return jsonify({"error": "No labels to export"}), 400

    try:
        # Group labels by sentence
        sentence_labels = defaultdict(list)
        for label in labeled_data:
            sentence_labels[label['sentence']].append({
                'role': label['role'],
                'text': label['text'],
                'word_indices': label['word_indices']
            })

        # Create rows for the DataFrame
        rows = []
        for sentence in current_texts:
            labels = sentence_labels[sentence]
            # Sort labels by the first word index in each label
            labels.sort(key=lambda x: min(x['word_indices']))
            
            # Create a row with the sentence and all its labels
            row = {'sentence': sentence}
            for i, label in enumerate(labels, 1):
                row[f'label_{i}_role'] = label['role']
                row[f'label_{i}_text'] = label['text']
                row[f'label_{i}_indices'] = ','.join(map(str, label['word_indices']))
            
            rows.append(row)

        # Create DataFrame and export to CSV
        df = pd.DataFrame(rows)
        
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(temp_dir, f'srl_labels_{timestamp}.csv')
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        return send_file(
            output_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'srl_labels_{timestamp}.csv'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 