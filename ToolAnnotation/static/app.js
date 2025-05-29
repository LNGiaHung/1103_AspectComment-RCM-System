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
        this.unsavedChanges = new Set();
        this.unsavedRowChanges = {};
    }

    reset() {
        this.currentPage = 0;
        this.totalPages = 0;
        this.totalRows = 0;
        this.selectedWords.clear();
        this.currentRowIndex = null;
        this.isLoading = false;
        this.currentData = [];
        this.unsavedChanges.clear();
        this.unsavedRowChanges = {};
    }

    addUnsavedChange(rowIndex) {
        this.unsavedChanges.add(rowIndex);
        this.updateUnsavedIndicator();
    }

    clearUnsavedChanges() {
        this.unsavedChanges.clear();
        this.updateUnsavedIndicator();
    }

    updateUnsavedIndicator() {
        const indicator = document.getElementById('unsavedIndicator');
        if (indicator) {
            indicator.style.display = this.unsavedChanges.size > 0 ? 'inline' : 'none';
            indicator.textContent = `${this.unsavedChanges.size} unsaved changes`;
        }
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
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    },

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
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
    },

    updateRowHeaderCounts(rowIndex) {
        // Find the row container by data-row-index
        const rowDiv = document.querySelector(`.row-container[data-row-index="${rowIndex}"]`);
        if (!rowDiv) return;
        const headerDiv = rowDiv.querySelector('.row-header');
        if (!headerDiv) return;

        // Get local unsaved changes if any, else count from DOM
        let aspectsCount = 0;
        let sentimentsCount = 0;
        if (state.unsavedRowChanges[rowIndex]) {
            aspectsCount = state.unsavedRowChanges[rowIndex].aspects?.length || 0;
            sentimentsCount = Object.values(state.unsavedRowChanges[rowIndex].sentiments || {}).filter(v => v).length;
        } else {
            aspectsCount = rowDiv.querySelectorAll('.word.aspect').length;
            sentimentsCount = rowDiv.querySelectorAll('.word.sentiment-positive, .word.sentiment-negative, .word.sentiment-neutral').length;
        }

        // Update the header text
        headerDiv.querySelector('small').innerHTML = `
            Row ${parseInt(rowIndex) + 1} | 
            Aspects: ${aspectsCount} | 
            Sentiments: ${sentimentsCount}
        `;
    }
};

// API functions with better error handling
const api = {
    async request(url, options = {}) {
        try {
            console.log(`Making request to ${url}`, options);
            utils.showLoading(true);
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });

            console.log('Response received:', response.status);
            const data = await response.json();
            console.log('Response data:', data);
            
            if (!response.ok) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }
            
            return data;
        } catch (error) {
            console.error(`API Error (${url}):`, error);
            utils.showAlert(`Error: ${error.message}`, 'danger');
            throw error;
        } finally {
            utils.showLoading(false);
        }
    },

    async uploadFile(formData) {
        try {
            console.log('Starting file upload...');
            utils.showLoading(true);
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            console.log('Upload response received:', response.status);
            const data = await response.json();
            console.log('Upload response data:', data);
            
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
        return this.request(`/get_data?page=${state.currentPage}`);
    },

    async updateData(rowIndex, aspects = null, sentiments = null) {
        const payload = { 
            rowIndex: rowIndex,
            modifications: {}
        };
        if (aspects !== null) payload.modifications.aspects = aspects;
        if (sentiments !== null) payload.modifications.sentiments = sentiments;
        
        try {
            const response = await this.request('/update_data', {
                method: 'POST',
                body: JSON.stringify(payload)
            });

            // Update the UI immediately after successful API call
            const rowElement = document.querySelector(`[data-row-index="${rowIndex}"]`);
            if (rowElement) {
                const wordContainer = rowElement.querySelector('.word-container');
                if (wordContainer) {
                    // Update aspects
                    if (aspects !== null) {
                        // Remove all existing aspect classes
                        wordContainer.querySelectorAll('.word').forEach(word => {
                            word.classList.remove('aspect');
                        });
                        // Add aspect class to new aspects
                        aspects.forEach(aspect => {
                            wordContainer.querySelectorAll('.word').forEach(word => {
                                if (word.textContent === aspect) {
                                    word.classList.add('aspect');
                                }
                            });
                        });
                    }

                    // Update sentiments
                    if (sentiments !== null) {
                        Object.entries(sentiments).forEach(([wordIndex, sentiment]) => {
                            const word = wordContainer.querySelector(`[data-word-index="${wordIndex}"]`);
                            if (word) {
                                word.classList.remove('sentiment-positive', 'sentiment-negative', 'sentiment-neutral');
                                if (sentiment) {
                                    word.classList.add(`sentiment-${sentiment}`);
                                }
                            }
                        });
                    }
                }
            }

            return response;
        } catch (error) {
            console.error('Error updating data:', error);
            throw error;
        }
    },

    async changePage(newPage) {
        try {
            clearSelection();
            // Save all unsaved changes for the current page
            const startIdx = state.currentPage * 10; // rows_per_page = 10
            const endIdx = startIdx + 10;
            const promises = [];
            for (let rowIdx = startIdx; rowIdx < endIdx; rowIdx++) {
                if (state.unsavedRowChanges[rowIdx]) {
                    const { aspects, sentiments } = state.unsavedRowChanges[rowIdx];
                    promises.push(this.updateData(Number(rowIdx), aspects, sentiments));
                    delete state.unsavedRowChanges[rowIdx];
                    state.unsavedChanges.delete(rowIdx);
                }
            }
            if (promises.length > 0) await Promise.all(promises);
            // Actually call the backend to change the page
            await this.request('/change_page', {
                method: 'POST',
                body: JSON.stringify({ page: newPage }),
                headers: { 'Content-Type': 'application/json' }
            });
            await loadData();
        } catch (error) {
            // Error already handled in api.changePage
        }
    },

    async saveData() {
        try {
            // Save all unsaved changes for all rows
            const promises = [];
            for (const rowIdx in state.unsavedRowChanges) {
                const { aspects, sentiments } = state.unsavedRowChanges[rowIdx];
                promises.push(this.updateData(Number(rowIdx), aspects, sentiments));
            }
            if (promises.length > 0) await Promise.all(promises);
            state.unsavedRowChanges = {};
            state.clearUnsavedChanges();
            const result = await this.saveData();
            let downloadLinks = '';
            if (result.csv_file) {
                downloadLinks += `<br><a href='/${result.csv_file}' download>Download Full CSV</a>`;
            }
            if (result.simplified_csv) {
                downloadLinks += `<br><a href='/${result.simplified_csv}' download>Download Simplified CSV</a>`;
            }
            utils.showAlert(
                `Annotations saved successfully!${downloadLinks}
                <br>Total rows: ${result.total_rows}
                <br>Annotated rows: ${result.annotated_rows}`,
                'success'
            );
        } catch (error) {
            // Error already handled in api.saveData
        }
    },

    async getStats() {
        return this.request('/get_stats');
    },

    async loadAutoSave() {
        return this.request('/load_auto_save', { method: 'POST' });
    }
};

// File Upload Handler
document.getElementById('uploadForm')?.addEventListener('submit', async function(e) {
    e.preventDefault(); // Prevent form submission
    console.log('Form submission started');
    
    const fileInput = document.getElementById('csvFile');
    const submitBtn = document.getElementById('uploadBtn');
    const uploadStatus = document.getElementById('uploadStatus');
    const file = fileInput.files[0];
    
    if (!file) {
        console.log('No file selected');
        utils.showAlert('Please select a file', 'warning');
        return;
    }

    console.log('File selected:', {
        name: file.name,
        type: file.type,
        size: file.size
    });

    if (!file.name.toLowerCase().endsWith('.csv')) {
        console.log('Invalid file type:', file.name);
        utils.showAlert('Please select a CSV file', 'warning');
        return;
    }

    if (file.size > 50 * 1024 * 1024) { // 50MB
        console.log('File too large:', file.size);
        utils.showAlert('File size must be less than 50MB', 'warning');
        return;
    }

    // Show loading state
    if (submitBtn) submitBtn.disabled = true;
    if (uploadStatus) uploadStatus.style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);
    console.log('FormData created with file');

    try {
        console.log('Sending upload request...');
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        console.log('Upload response received:', response.status);
        const data = await response.json();
        console.log('Upload response data:', data);
        
        if (!response.ok) {
            throw new Error(data.error || 'Upload failed');
        }
        
        utils.showAlert(data.message, 'success');
        state.reset();
        state.totalRows = data.total_rows;
        
        document.getElementById('dataSection').style.display = 'block';
        await loadData();
        await updateStats();
        
        // Clear the file input
        fileInput.value = '';
        
    } catch (error) {
        console.error('Upload error:', error);
        utils.showAlert(`Upload failed: ${error.message}`, 'danger');
    } finally {
        if (submitBtn) submitBtn.disabled = false;
        if (uploadStatus) uploadStatus.style.display = 'none';
    }
});

// Data loading and display
async function loadData() {
    try {
        const data = await api.getData();
        state.currentData = data.data;
        displayData(data.data);
        updateNavigation(data);
        updateStats();
    } catch (error) {
        // Error already handled in api.getData
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
        
        // Add row header with info
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
            
            // Apply existing annotations
            if (Array.isArray(row.aspects) && row.aspects.includes(word)) {
                wordSpan.classList.add('aspect');
            }
            
            if (typeof row.sentiments === 'object' && row.sentiments[wordIndex]) {
                wordSpan.classList.add(`sentiment-${row.sentiments[wordIndex]}`);
            }
            
            // Add click handler
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
    
    // Update page info
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
    
    // Update button states
    const prevBtn = document.getElementById('prevPage');
    const nextBtn = document.getElementById('nextPage');
    
    if (prevBtn) prevBtn.disabled = !data.has_prev;
    if (nextBtn) nextBtn.disabled = !data.has_next;
    
    // Update state
    state.currentPage = data.current_page;
    state.totalPages = data.total_pages;
    state.totalRows = data.total_rows;
}

// Page navigation handlers
document.getElementById('prevPage')?.addEventListener('click', async () => {
    if (state.currentPage > 0 && !state.isLoading) {
        await changePage(state.currentPage - 1);
    }
});

document.getElementById('nextPage')?.addEventListener('click', async () => {
    if (state.currentPage < state.totalPages - 1 && !state.isLoading) {
        await changePage(state.currentPage + 1);
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
    
    // Check if all words are from the same row
    const rowIndices = [...new Set(words.map(w => w.rowIndex))];
    if (rowIndices.length > 1) {
        utils.showAlert('Can only combine words from the same row', 'warning');
        return;
    }
    
    // Sort by word index
    words.sort((a, b) => a.wordIndex - b.wordIndex);
    
    const firstWord = words[0].element;
    const combinedText = words.map(w => w.text).join(' ');
    
    // Update the first word with combined text
    firstWord.textContent = combinedText;
    firstWord.classList.remove('selected');
    
    // Remove other words
    words.slice(1).forEach(w => w.element.remove());
    
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

// Annotation functions
document.getElementById('markAspectBtn')?.addEventListener('click', async () => {
    if (!utils.validateWordSelection()) return;
    const wordsData = utils.getSelectedWordsData();
    const rowIndices = [...new Set(wordsData.map(w => w.rowIndex))];
    if (rowIndices.length > 1) {
        utils.showAlert('Can only mark aspects from the same row', 'warning');
        return;
    }
    const aspects = wordsData.map(w => w.text);
    const actualRowIndex = parseInt(wordsData[0].element.dataset.actualRowIndex);
    // Update UI
    wordsData.forEach(w => {
        w.element.classList.add('aspect');
        w.element.classList.remove('selected');
    });
    // Update local cache
    if (!state.unsavedRowChanges[actualRowIndex]) state.unsavedRowChanges[actualRowIndex] = { aspects: [], sentiments: {} };
    state.unsavedRowChanges[actualRowIndex].aspects = aspects;
    state.addUnsavedChange(actualRowIndex);
    utils.updateRowHeaderCounts(actualRowIndex);
    clearSelection();
    utils.showAlert(`Marked ${aspects.length} words as aspects (not yet saved)`, 'success');
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
    // Update UI
    wordsData.forEach(w => {
        w.element.classList.remove('sentiment-positive', 'sentiment-negative', 'sentiment-neutral');
        w.element.classList.add(`sentiment-${sentiment}`);
        w.element.classList.remove('selected');
    });
    // Update local cache
    if (!state.unsavedRowChanges[actualRowIndex]) state.unsavedRowChanges[actualRowIndex] = { aspects: [], sentiments: {} };
    state.unsavedRowChanges[actualRowIndex].sentiments = {
        ...state.unsavedRowChanges[actualRowIndex].sentiments,
        ...sentiments
    };
    state.addUnsavedChange(actualRowIndex);
    utils.updateRowHeaderCounts(actualRowIndex);
    clearSelection();
    utils.showAlert(`Applied ${sentiment} sentiment (not yet saved)`, 'success');
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
    // Update UI
    wordsData.forEach(w => {
        w.element.classList.remove('aspect', 'sentiment-positive', 'sentiment-negative', 'sentiment-neutral', 'selected');
    });
    // Update local cache
    if (!state.unsavedRowChanges[actualRowIndex]) state.unsavedRowChanges[actualRowIndex] = { aspects: [], sentiments: {} };
    // Remove aspects and sentiments for selected words
    state.unsavedRowChanges[actualRowIndex].aspects = [];
    wordsData.forEach(w => {
        state.unsavedRowChanges[actualRowIndex].sentiments[w.wordIndex] = null;
    });
    state.addUnsavedChange(actualRowIndex);
    utils.updateRowHeaderCounts(actualRowIndex);
    clearSelection();
    utils.showAlert('Annotations cleared (not yet saved)', 'success');
});

// Save functionality
document.getElementById('saveBtn')?.addEventListener('click', async () => {
    if (!confirm('Save all annotations? This will create a new CSV file with current annotations.')) {
        return;
    }
    try {
        // Only call saveData, do not send cached updates here
        const result = await api.saveData();
        let downloadLinks = '';
        if (result.csv_file) {
            downloadLinks += `<br><a href='/${result.csv_file}' download>Download Full CSV</a>`;
        }
        if (result.simplified_csv) {
            downloadLinks += `<br><a href='/${result.simplified_csv}' download>Download Simplified CSV</a>`;
        }
        utils.showAlert(
            `Annotations saved successfully!${downloadLinks}
            <br>Total rows: ${result.total_rows}
            <br>Annotated rows: ${result.annotated_rows}`,
            'success'
        );
        state.clearUnsavedChanges();
        state.unsavedRowChanges = {};
    } catch (error) {
        // Error already handled in api.saveData
    }
});

// Statistics and progress tracking
async function updateStats() {
    try {
        const stats = await api.getStats();
        
        const statsContainer = document.getElementById('statsContainer');
        if (statsContainer) {
            statsContainer.innerHTML = `
                <div class="row">
                    <div class="col-md-3">
                        <div class="stat-card">
                            <h6>Progress</h6>
                            <div class="progress mb-2">
                                <div class="progress-bar" style="width: ${stats.completion_percentage}%"></div>
                            </div>
                            <small>${stats.completion_percentage}% complete</small>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="stat-card">
                            <h6>Rows</h6>
                            <div>${stats.annotated_rows}/${stats.total_rows}</div>
                            <small>annotated</small>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="stat-card">
                            <h6>Aspects</h6>
                            <div>${stats.total_aspects}</div>
                            <small>identified</small>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="stat-card">
                            <h6>Sentiments</h6>
                            <div>${stats.total_sentiments}</div>
                            <small>labeled</small>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="stat-card">
                            <h6>Sentiment Distribution</h6>
                            <div class="sentiment-dist">
                                <span class="badge bg-success">+${stats.sentiment_distribution.positive}</span>
                                <span class="badge bg-danger">-${stats.sentiment_distribution.negative}</span>
                                <span class="badge bg-secondary">~${stats.sentiment_distribution.neutral}</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
    } catch (error) {
        console.error('Failed to update stats:', error);
    }
}

// Auto-save functionality
document.getElementById('loadAutoSaveBtn')?.addEventListener('click', async () => {
    if (!confirm('Load auto-saved data? This will replace current data.')) {
        return;
    }
    
    try {
        const result = await api.loadAutoSave();
        state.reset();
        state.totalRows = result.total_rows;
        
        document.getElementById('dataSection').style.display = 'block';
        await loadData();
        await updateStats();
        
        utils.showAlert(result.message, 'success');
        
    } catch (error) {
        // Error already handled in api.loadAutoSave
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
            case 's':
                e.preventDefault();
                document.getElementById('saveBtn')?.click();
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

// Prevent page unload with unsaved changes
window.addEventListener('beforeunload', (e) => {
    if (state.unsavedChanges.size > 0) {
        e.preventDefault();
        e.returnValue = 'You have unsaved changes. Are you sure you want to leave?';
    }
});

// Initialize tooltips and other UI enhancements
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Bootstrap tooltips if available
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    // Add selection info display
    const dataSection = document.getElementById('dataSection');
    if (dataSection) {
        const selectionInfo = document.createElement('div');
        selectionInfo.id = 'selectionInfo';
        selectionInfo.className = 'alert alert-info mt-2';
        selectionInfo.textContent = 'No words selected';
        dataSection.insertBefore(selectionInfo, document.getElementById('dataContainer'));
    }
    
    // Add unsaved changes indicator
    const saveBtn = document.getElementById('saveBtn');
    if (saveBtn) {
        const indicator = document.createElement('span');
        indicator.id = 'unsavedIndicator';
        indicator.className = 'badge bg-warning ms-2';
        indicator.style.display = 'none';
        saveBtn.parentNode.insertBefore(indicator, saveBtn.nextSibling);
    }
});

// Export for potential external use
window.AnnotationTool = {
    state,
    api,
    utils,
    loadData,
    clearSelection,
    updateStats
};