# Semantic Role Labeling Tool

## Overview

This tool is a web-based application for annotating text data with semantic roles (such as Agent, Patient, Theme, etc.). It is designed to help users label sentences or text snippets for tasks in Natural Language Processing (NLP), especially for Semantic Role Labeling (SRL) datasets.

## How the Tool Works

1. **Upload CSV File**:
   - Click the **Upload** button and select a CSV file containing the texts you want to label. Each row should contain a text snippet in the first column.
2. **Navigate Texts**:
   - Use the **Previous** and **Next** buttons to move through the texts loaded from your CSV file. The current text is displayed in the center panel.
3. **Labeling**:
   - For each text, select a semantic role from the dropdown menu (e.g., Agent, Patient, Theme, etc.).
   - Click **Save Label** to assign the selected role to the current text. The label will appear in the "Current Labels" panel on the right.
4. **Export Labels**:
   - Once you have finished labeling, click **Export to CSV** to download a CSV file containing the original texts and their assigned labels.

## Expected Output After Labeling

The exported CSV contains detailed information for each labeled span in the sentence. The columns are as follows:

- `sentence`: The original sentence.
- For each label (e.g., label_1, label_2, ...):
  - `label_X_r`: The semantic role (e.g., agent, manner, time, source).
  - `label_X_te`: The text span in the sentence corresponding to the label.
  - `label_X_ir`: The index or range (e.g., token indices) of the labeled span in the sentence.

### Example Output

| sentence                                | label_1_r   | label_1_te     | label_1_ir | label_2_r   | label_2_te | label_2_ir | label_3_r | label_3_te    | label_3_ir |
| --------------------------------------- | ----------- | -------------- | ---------- | ----------- | ---------- | ---------- | --------- | ------------- | ---------- |
| I loved the acting in this movie.       | experiencer | I              | 0          | theme       | the acting | 2,3        | location  | in this movie | 4,5,6      |
| The plot was confusing but interesting. | theme       | The plot       | 0,1        | manner      | confusing  | 3          | manner    | interesting   | 5          |
| The movie made me cry.                  | theme       | The movie      | 0,1        | experiencer | me         | 3          | result    | cry           | 4          |
| The soundtrack was amazing.             | theme       | The soundtrack | 0,1,2      | manner      | amazing    | 4          |           |               |            |

- Each `label_X_r`/`label_X_te`/`label_X_ir` group corresponds to a labeled span in the sentence.
- The number of label columns depends on how many spans were labeled in each sentence.

This output can be used for training or evaluating NLP models for semantic role labeling tasks, supporting multiple roles and spans per sentence.

## Requirements

- Python 3.x
- Flask
- (See `requirements.txt` for full dependencies)

## Running the Tool

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the application:
   ```bash
   python app.py
   ```
3. Open your browser and go to `http://localhost:5000` to use the tool.

---

For any issues or questions, please contact the maintainer.
