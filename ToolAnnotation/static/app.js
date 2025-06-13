// Global state management
class AnnotationState {
    constructor() {
        this.currentPage = 0;
        this.totalPages = 0;
        this.totalRows = 0;
        this.selectedWords = new Set();
        this.currentRowIndex = null;
        this.isLoading = false;
        this.currentData = [];
        this.annotatedData = []; // Store complete annotated data
    }

    reset() {
        this.currentPage = 0;
        this.totalPages = 0;
        this.totalRows = 0;
        this.selectedWords.clear();
        this.currentRowIndex = null;
        this.isLoading = false;
        this.currentData = [];
        this.annotatedData = [];
    }

    updateAnnotatedData(rowIndex, aspects, sentiments) {
        if (!this.annotatedData[rowIndex]) {
            this.annotatedData[rowIndex] = {
                id: rowIndex,
                text: this.currentData[rowIndex]?.text || '',
                aspects: [],
                sentiments: {},
                is_annotated: false
            };
        }
        
        if (aspects) {
            this.annotatedData[rowIndex].aspects = aspects;
        }
        if (sentiments) {
            this.annotatedData[rowIndex].sentiments = {
                ...this.annotatedData[rowIndex].sentiments,
                ...sentiments
            };
        }
        
        this.annotatedData[rowIndex].is_annotated = 
            this.annotatedData[rowIndex].aspects.length > 0 || 
            Object.values(this.annotatedData[rowIndex].sentiments).some(v => v);
    }
}

const state = new AnnotationState();

// Utility functions
const utils = {
    showLoading(show = true) {
        const spinner = document.getElementById('loadingSpinner');
        if (spinner) {
            spinner.style.display = show ? 'block' : 'none';
        }
        state.isLoading = show;
    },

    showAlert(message, type = 'info') {
        const alertContainer = document.getElementById('alertContainer');
        if (!alertContainer) return;

        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        alertContainer.appendChild(alertDiv);
        
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    },

    validateWordSelection() {
        if (state.selectedWords.size === 0) {
            this.showAlert('Please select at least one word', 'warning');
            return false;
        }
        return true;
    },

    getSelectedWordsData() {
        return Array.from(state.selectedWords).map(word => ({
            element: word,
            text: word.textContent,
            rowIndex: parseInt(word.dataset.actualRowIndex),
            wordIndex: parseInt(word.dataset.wordIndex)
        }));
    }
};

// API functions
const api = {
    async request(url, options = {}) {
        try {
            console.log(`Making request to ${url}`, options);
            utils.showLoading(true);
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 seconds timeout
            
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                signal: controller.signal,
                ...options
            });

            clearTimeout(timeoutId);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }
            
            return data;
        } catch (error) {
            console.error(`API Error (${url}):`, error);
            if (error.name === 'AbortError') {
                utils.showAlert('Request timed out. Please try again.', 'danger');
            } else {
                utils.showAlert(`Error: ${error.message}`, 'danger');
            }
            throw error;
        } finally {
            utils.showLoading(false);
        }
    },

    async uploadFile(formData) {
        try {
            utils.showLoading(true);
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Upload failed');
            }
            
            return data;
        } catch (error) {
            console.error('Upload Error:', error);
            utils.showAlert(`Upload failed: ${error.message}`, 'danger');
            throw error;
        } finally {
            utils.showLoading(false);
        }
    },

    async getData() {
        try {
            const data = await this.request(`/get_data?page=${state.currentPage}`);
            
            if (!data || !Array.isArray(data.data)) {
                throw new Error('Invalid data format received from server');
            }
            
            state.currentData = data.data;
            
            data.data.forEach((row, index) => {
                if (!state.annotatedData[row.row_index]) {
                    state.annotatedData[row.row_index] = {
                        id: row.row_index,
                        text: row.text,
                        aspects: row.aspects || [],
                        sentiments: row.sentiments || {},
                        is_annotated: row.is_annotated || false
                    };
                }
            });
            
            return data;
        } catch (error) {
            console.error('Error getting data:', error);
            throw error;
        }
    },

    async updateData(rowIndex, aspects = null, sentiments = null) {
        if (aspects !== null && !Array.isArray(aspects)) {
            throw new Error('Aspects must be an array');
        }
        if (sentiments !== null && typeof sentiments !== 'object') {
            throw new Error('Sentiments must be an object');
        }

        const payload = { 
            rowIndex: parseInt(rowIndex),
            modifications: {}
        };

        if (aspects !== null) {
            payload.modifications.aspects = aspects;
        }
        if (sentiments !== null) {
            payload.modifications.sentiments = sentiments;
        }
        
        try {
            const response = await this.request('/update_data', {
                method: 'POST',
                body: JSON.stringify(payload)
            });

            // Update local state
            state.updateAnnotatedData(rowIndex, aspects, sentiments);
            
            return response;
        } catch (error) {
            console.error('Error updating data:', error);
            throw error;
        }
    },

    async changePage(newPage) {
        try {
            clearSelection();
            await this.request('/change_page', {
                method: 'POST',
                body: JSON.stringify({ page: newPage })
            });
            await loadData();
        } catch (error) {
            console.error('Error changing page:', error);
        }
    },

    async exportCsv() {
        try {
            const result = await this.request('/export_csv', {
                method: 'POST',
                body: JSON.stringify({
                    annotated_data: state.annotatedData.filter(d => d && d.is_annotated)
                })
            });
            
            let downloadLink = '';
            if (result.csv_file) {
                downloadLink = `<br><a href='/${result.csv_file}' download>Download CSV File</a>`;
            }
            
            utils.showAlert(
                `Annotations exported successfully!${downloadLink}
                <br>Total rows: ${result.total_rows}
                <br>Annotated rows: ${result.annotated_rows}`,
                'success'
            );
            return result;
        } catch (error) {
            console.error('Error exporting CSV:', error);
            throw error;
        }
    }
};

// File Upload Handler
document.getElementById('uploadForm')?.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    if (state.isLoading) {
        utils.showAlert('Upload already in progress. Please wait.', 'warning');
        return;
    }
    
    const fileInput = document.getElementById('csvFile');
    const submitBtn = document.getElementById('uploadBtn');
    const file = fileInput.files[0];
    
    if (!file) {
        utils.showAlert('Please select a file', 'warning');
        return;
    }

    if (!file.name.toLowerCase().endsWith('.csv')) {
        utils.showAlert('Please select a CSV file', 'warning');
        return;
    }

    if (file.size > 50 * 1024 * 1024) { // 50MB
        utils.showAlert('File size must be less than 50MB', 'warning');
        return;
    }

    if (submitBtn) submitBtn.disabled = true;
    utils.showLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const data = await api.uploadFile(formData);
        utils.showAlert(data.message, 'success');
        state.reset();
        state.totalRows = data.total_rows;
        
        document.getElementById('dataSection').style.display = 'block';
        await loadData();
        
        fileInput.value = '';
    } catch (error) {
        console.error('Upload error:', error);
    } finally {
        if (submitBtn) submitBtn.disabled = false;
        utils.showLoading(false);
    }
});

// Data loading and display
async function loadData() {
    try {
        const data = await api.getData();
        state.currentData = data.data;
        displayData(data.data);
        updateNavigation(data);
    } catch (error) {
        console.error('Error loading data:', error);
    }
}

function displayData(rows) {
    const container = document.getElementById('dataContainer');
    if (!container) return;
    
    container.innerHTML = '';
    
    if (!rows || rows.length === 0) {
        container.innerHTML = '<div class="alert alert-info">No data to display</div>';
        return;
    }
    
    rows.forEach((row, rowIndex) => {
        const rowDiv = document.createElement('div');
        rowDiv.className = 'row-container mb-4 p-3 border rounded';
        rowDiv.dataset.rowIndex = rowIndex;
        
        const headerDiv = document.createElement('div');
        headerDiv.className = 'row-header mb-2 d-flex justify-content-between align-items-center';
        headerDiv.innerHTML = `
            <div>
                <small class="text-muted">
                    Row ${row.row_index + 1} | 
                    Aspects: ${Array.isArray(row.aspects) ? row.aspects.length : 0} | 
                    Sentiments: ${typeof row.sentiments === 'object' ? Object.keys(row.sentiments).length : 0}
                </small>
            </div>
            <div>
                <span class="badge ${row.is_annotated ? 'bg-success' : 'bg-warning'}">
                    ${row.is_annotated ? 'Annotated' : 'Not Annotated'}
                </span>
            </div>
        `;
        
        const words = row.text.split(/(\s+)/).filter(word => word.trim().length > 0);
        const wordContainer = document.createElement('div');
        wordContainer.className = 'word-container';
        
        words.forEach((word, wordIndex) => {
            const wordSpan = document.createElement('span');
            wordSpan.className = 'word';
            wordSpan.textContent = word;
            wordSpan.dataset.rowIndex = rowIndex;
            wordSpan.dataset.wordIndex = wordIndex;
            wordSpan.dataset.actualRowIndex = row.row_index;
            
            // Add aspect styling
            if (Array.isArray(row.aspects) && row.aspects.includes(word)) {
                wordSpan.classList.add('aspect');
                wordSpan.style.borderBottom = '2px solid #007bff';
                wordSpan.style.fontWeight = 'bold';
            }
            
            // Add sentiment styling
            if (typeof row.sentiments === 'object' && row.sentiments[wordIndex]) {
                const sentiment = row.sentiments[wordIndex];
                wordSpan.classList.add(`sentiment-${sentiment}`);
                
                // Add sentiment-specific styling
                switch(sentiment) {
                    case 'positive':
                        wordSpan.style.backgroundColor = 'rgba(40, 167, 69, 0.2)';
                        wordSpan.style.borderLeft = '3px solid #28a745';
                        break;
                    case 'negative':
                        wordSpan.style.backgroundColor = 'rgba(220, 53, 69, 0.2)';
                        wordSpan.style.borderLeft = '3px solid #dc3545';
                        break;
                    case 'neutral':
                        wordSpan.style.backgroundColor = 'rgba(108, 117, 125, 0.2)';
                        wordSpan.style.borderLeft = '3px solid #6c757d';
                        break;
                }
            }
            
            wordSpan.addEventListener('click', (e) => {
                e.stopPropagation();
                toggleWordSelection(wordSpan);
            });
            
            wordContainer.appendChild(wordSpan);
        });
        
        rowDiv.appendChild(headerDiv);
        rowDiv.appendChild(wordContainer);
        container.appendChild(rowDiv);
    });
}

function toggleWordSelection(wordSpan) {
    const isSelected = wordSpan.classList.contains('selected');
    
    if (isSelected) {
        wordSpan.classList.remove('selected');
        state.selectedWords.delete(wordSpan);
    } else {
        wordSpan.classList.add('selected');
        state.selectedWords.add(wordSpan);
        state.currentRowIndex = parseInt(wordSpan.dataset.actualRowIndex);
    }
    
    updateSelectionInfo();
}

function updateSelectionInfo() {
    const info = document.getElementById('selectionInfo');
    if (info) {
        const count = state.selectedWords.size;
        const selectedWords = Array.from(state.selectedWords).map(w => w.textContent).join(', ');
        info.textContent = count > 0 ? `Selected: ${selectedWords} (${count} words)` : 'No words selected';
    }
}

function clearSelection() {
    state.selectedWords.forEach(word => {
        word.classList.remove('selected');
    });
    state.selectedWords.clear();
    state.currentRowIndex = null;
    updateSelectionInfo();
}

// Navigation functions
function updateNavigation(data) {
    if (!data) return;
    
    const elements = {
        currentPage: data.current_page + 1,
        totalPages: data.total_pages,
        startIndex: data.start_idx,
        endIndex: data.end_idx,
        totalRows: data.total_rows
    };
    
    Object.entries(elements).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) element.textContent = value;
    });
    
    const prevBtn = document.getElementById('prevPage');
    const nextBtn = document.getElementById('nextPage');
    
    if (prevBtn) prevBtn.disabled = !data.has_prev;
    if (nextBtn) nextBtn.disabled = !data.has_next;
    
    state.currentPage = data.current_page;
    state.totalPages = data.total_pages;
    state.totalRows = data.total_rows;
}

// Page navigation handlers
document.getElementById('prevPage')?.addEventListener('click', async () => {
    if (state.currentPage > 0 && !state.isLoading) {
        await api.changePage(state.currentPage - 1);
    }
});

document.getElementById('nextPage')?.addEventListener('click', async () => {
    if (state.currentPage < state.totalPages - 1 && !state.isLoading) {
        await api.changePage(state.currentPage + 1);
    }
});

// Annotation functions
document.getElementById('markAspectBtn')?.addEventListener('click', async () => {
    if (!utils.validateWordSelection()) return;
    const wordsData = utils.getSelectedWordsData();
    console.log('[MARK ASPECT] Selected words data:', wordsData);
    const rowIndices = [...new Set(wordsData.map(w => w.rowIndex))];
    console.log('[MARK ASPECT] Row indices:', rowIndices);
    if (rowIndices.length > 1) {
        utils.showAlert('Can only mark aspects from the same row', 'warning');
        return;
    }
    
    // Get the actual text content of selected words, preserving spaces in combined words
    const aspects = wordsData.map(w => w.element.textContent.trim());
    const actualRowIndex = parseInt(wordsData[0].element.dataset.actualRowIndex);
    console.log('[MARK ASPECT] Aspects to mark:', aspects);
    
    // Get existing aspects for this row
    const existingAspects = state.annotatedData[actualRowIndex]?.aspects || [];
    console.log('[MARK ASPECT] Existing aspects:', existingAspects);
    
    // Combine with existing aspects, avoiding duplicates
    const updatedAspects = [...new Set([...existingAspects, ...aspects])];
    console.log('[MARK ASPECT] Updated aspects:', updatedAspects);
    
    await api.updateData(actualRowIndex, updatedAspects, null);
    clearSelection();
    await loadData();
    utils.showAlert(`Marked ${aspects.length} words as aspects`, 'success');
});

document.getElementById('applySentimentBtn')?.addEventListener('click', async () => {
    const sentiment = document.getElementById('sentimentSelect')?.value;
    if (!sentiment) {
        utils.showAlert('Please select a sentiment', 'warning');
        return;
    }
    if (!utils.validateWordSelection()) return;
    const wordsData = utils.getSelectedWordsData();
    const rowIndices = [...new Set(wordsData.map(w => w.rowIndex))];
    if (rowIndices.length > 1) {
        utils.showAlert('Can only apply sentiment to words from the same row', 'warning');
        return;
    }
    const sentiments = {};
    wordsData.forEach(w => {
        sentiments[w.wordIndex] = sentiment;
    });
    const actualRowIndex = parseInt(wordsData[0].element.dataset.actualRowIndex);
    
    await api.updateData(actualRowIndex, null, sentiments);
    clearSelection();
    await loadData();
    utils.showAlert(`Applied ${sentiment} sentiment`, 'success');
});

document.getElementById('clearAnnotationsBtn')?.addEventListener('click', async () => {
    if (!utils.validateWordSelection()) return;
    if (!confirm('Are you sure you want to clear annotations for selected words?')) {
        return;
    }
    const wordsData = utils.getSelectedWordsData();
    const rowIndices = [...new Set(wordsData.map(w => w.rowIndex))];
    if (rowIndices.length > 1) {
        utils.showAlert('Can only clear annotations from the same row', 'warning');
        return;
    }
    const actualRowIndex = parseInt(wordsData[0].element.dataset.actualRowIndex);
    
    wordsData.forEach(w => {
        w.element.classList.remove('aspect', 'sentiment-positive', 'sentiment-negative', 'sentiment-neutral', 'selected');
    });
    
    await api.updateData(actualRowIndex, [], {});
    clearSelection();
    await loadData();
    utils.showAlert('Annotations cleared', 'success');
});

// Export functionality
document.getElementById('exportCsvBtn')?.addEventListener('click', async () => {
    if (!confirm('Export annotations to CSV?')) {
        return;
    }
    try {
        await api.exportCsv();
    } catch (error) {
        console.error('Error exporting CSV:', error);
    }
});

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Add selection info display
    const dataSection = document.getElementById('dataSection');
    if (dataSection) {
        const selectionInfo = document.createElement('div');
        selectionInfo.id = 'selectionInfo';
        selectionInfo.className = 'alert alert-info mt-2';
        selectionInfo.textContent = 'No words selected';
        dataSection.insertBefore(selectionInfo, document.getElementById('dataContainer'));
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
            case 'e':
                e.preventDefault();
                document.getElementById('exportCsvBtn')?.click();
                break;
            case 'a':
                e.preventDefault();
                document.getElementById('markAspectBtn')?.click();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                document.getElementById('prevPage')?.click();
                break;
            case 'ArrowRight':
                e.preventDefault();
                document.getElementById('nextPage')?.click();
                break;
        }
    }
    
    if (e.key === 'Escape') {
        clearSelection();
    }
});

// Word manipulation functions
document.getElementById('combineBtn')?.addEventListener('click', async () => {
    if (!utils.validateWordSelection()) return;
    
    if (state.selectedWords.size < 2) {
        utils.showAlert('Please select at least 2 words to combine', 'warning');
        return;
    }
    
    const words = utils.getSelectedWordsData();
    console.log('[COMBINE] Selected words data:', words);
    
    // Check if all words are from the same row
    const rowIndices = [...new Set(words.map(w => w.rowIndex))];
    console.log('[COMBINE] Row indices:', rowIndices);
    if (rowIndices.length > 1) {
        utils.showAlert('Can only combine words from the same row', 'warning');
        return;
    }
    
    // Sort by word index
    words.sort((a, b) => a.wordIndex - b.wordIndex);
    console.log('[COMBINE] Sorted words:', words);
    
    const firstWord = words[0].element;
    const combinedText = words.map(w => w.text).join(' ');
    const actualRowIndex = parseInt(firstWord.dataset.actualRowIndex);
    console.log('[COMBINE] Combined text:', combinedText);
    
    // Get the container and all words in the row
    const container = firstWord.parentElement;
    const allWords = Array.from(container.children);
    console.log('[COMBINE] All words in row before combine:', allWords.map(w => w.textContent));
    
    // Find the indices of the words to combine
    const startIdx = allWords.indexOf(firstWord);
    const endIdx = allWords.indexOf(words[words.length - 1].element);
    console.log('[COMBINE] StartIdx:', startIdx, 'EndIdx:', endIdx);
    
    // Update the first word with combined text and preserve its data attributes
    firstWord.textContent = combinedText;
    firstWord.classList.remove('selected');
    
    // Remove the other words
    for (let i = endIdx; i > startIdx; i--) {
        allWords[i].remove();
    }
    
    // Update word indices for all remaining words
    const updatedWords = Array.from(container.children);
    updatedWords.forEach((word, idx) => {
        if (word.dataset.wordIndex) {
            word.dataset.wordIndex = idx;
        }
    });
    console.log('[COMBINE] All words in row after combine:', updatedWords.map(w => w.textContent));
    
    clearSelection();
    utils.showAlert('Words combined successfully', 'success');
});

document.getElementById('separateBtn')?.addEventListener('click', async () => {
    if (!utils.validateWordSelection()) return;
    
    if (state.selectedWords.size !== 1) {
        utils.showAlert('Please select exactly one word to separate', 'warning');
        return;
    }
    
    const wordData = utils.getSelectedWordsData()[0];
    const word = wordData.element;
    const text = wordData.text.trim();
    
    // Split by whitespace
    const separatedWords = text.split(/\s+/).filter(w => w.length > 0);
    
    if (separatedWords.length <= 1) {
        utils.showAlert('This word cannot be separated further', 'warning');
        return;
    }
    
    const container = word.parentElement;
    const wordIndex = Array.from(container.children).indexOf(word);
    
    // Create new word elements
    separatedWords.forEach((w, i) => {
        const newWord = document.createElement('span');
        newWord.className = 'word';
        newWord.textContent = w;
        newWord.dataset.rowIndex = word.dataset.rowIndex;
        newWord.dataset.wordIndex = wordIndex + i;
        newWord.dataset.actualRowIndex = word.dataset.actualRowIndex;
        
        newWord.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleWordSelection(newWord);
        });
        
        container.insertBefore(newWord, word);
    });
    
    word.remove();
    clearSelection();
    utils.showAlert('Word separated successfully', 'success');
});

// Add CSS styles to the document
const style = document.createElement('style');
style.textContent = `
    .word {
        display: inline-block;
        padding: 2px 4px;
        margin: 2px;
        border-radius: 3px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .word:hover {
        background-color: rgba(0, 123, 255, 0.1);
    }
    
    .word.selected {
        background-color: rgba(0, 123, 255, 0.2);
        outline: 2px solid #007bff;
    }
    
    .word.aspect {
        border-bottom: 2px solid #007bff;
        font-weight: bold;
    }
    
    .word.sentiment-positive {
        background-color: rgba(40, 167, 69, 0.2);
        border-left: 3px solid #28a745;
    }
    
    .word.sentiment-negative {
        background-color: rgba(220, 53, 69, 0.2);
        border-left: 3px solid #dc3545;
    }
    
    .word.sentiment-neutral {
        background-color: rgba(108, 117, 125, 0.2);
        border-left: 3px solid #6c757d;
    }
    
    .word-container {
        line-height: 2;
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 4px;
    }
`;
document.head.appendChild(style);