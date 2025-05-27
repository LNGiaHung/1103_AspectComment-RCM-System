let currentPage = 0;
let totalPages = 0;
let selectedWords = new Set();
let currentRowIndex = null;

// File Upload Handler
document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('csvFile');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (response.ok) {
            document.getElementById('uploadStatus').innerHTML = `
                <div class="alert alert-success">${data.message}</div>
            `;
            document.getElementById('dataSection').style.display = 'block';
            loadData();
        } else {
            document.getElementById('uploadStatus').innerHTML = `
                <div class="alert alert-danger">${data.error}</div>
            `;
        }
    } catch (error) {
        document.getElementById('uploadStatus').innerHTML = `
            <div class="alert alert-danger">Error uploading file</div>
        `;
    }
});

// Load Data
async function loadData() {
    try {
        const response = await fetch(`/get_data?page=${currentPage}`);
        const data = await response.json();
        
        if (response.ok) {
            displayData(data.data);
            updateNavigation(data);
        }
    } catch (error) {
        console.error('Error loading data:', error);
    }
}

// Update Navigation
function updateNavigation(data) {
    // Update page numbers
    document.getElementById('currentPage').textContent = data.current_page + 1;
    document.getElementById('totalPages').textContent = data.total_pages;
    document.getElementById('startIndex').textContent = data.start_idx;
    document.getElementById('endIndex').textContent = data.end_idx;
    document.getElementById('totalRows').textContent = data.total_rows;

    // Update button states
    document.getElementById('prevPage').disabled = !data.has_prev;
    document.getElementById('nextPage').disabled = !data.has_next;
}

// Navigation Button Handlers
document.getElementById('prevPage').addEventListener('click', async () => {
    if (currentPage > 0) {
        await changePage(currentPage - 1);
    }
});

document.getElementById('nextPage').addEventListener('click', async () => {
    if (currentPage < totalPages - 1) {
        await changePage(currentPage + 1);
    }
});

// Change Page with Auto-save
async function changePage(newPage) {
    try {
        const response = await fetch('/change_page', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                page: newPage
            })
        });

        if (response.ok) {
            currentPage = newPage;
            await loadData();
        } else {
            const data = await response.json();
            alert(data.error || 'Error changing page');
        }
    } catch (error) {
        console.error('Error changing page:', error);
        alert('Error changing page');
    }
}

// Display Data
function displayData(rows) {
    const container = document.getElementById('dataContainer');
    container.innerHTML = '';
    
    rows.forEach((row, rowIndex) => {
        const rowDiv = document.createElement('div');
        rowDiv.className = 'mb-4';
        
        const words = row.text.split(' ');
        const wordContainer = document.createElement('div');
        wordContainer.className = 'word-container';
        
        words.forEach((word, wordIndex) => {
            const wordSpan = document.createElement('span');
            wordSpan.className = 'word';
            wordSpan.textContent = word;
            wordSpan.dataset.rowIndex = rowIndex;
            wordSpan.dataset.wordIndex = wordIndex;
            
            // Check if this word is an aspect
            if (row.aspects && row.aspects.includes(word)) {
                wordSpan.classList.add('aspect');
            }
            
            // Check if this word has a sentiment
            if (row.sentiments && row.sentiments[wordIndex]) {
                wordSpan.classList.add(`sentiment-${row.sentiments[wordIndex]}`);
            }
            
            wordSpan.addEventListener('click', () => {
                wordSpan.classList.toggle('selected');
                if (wordSpan.classList.contains('selected')) {
                    selectedWords.add(wordSpan);
                    currentRowIndex = rowIndex;
                } else {
                    selectedWords.delete(wordSpan);
                }
            });
            
            wordContainer.appendChild(wordSpan);
        });
        
        rowDiv.appendChild(wordContainer);
        container.appendChild(rowDiv);
    });
}

// Combine Words
document.getElementById('combineBtn').addEventListener('click', () => {
    if (selectedWords.size < 2) {
        alert('Please select at least 2 words to combine');
        return;
    }
    
    const words = Array.from(selectedWords);
    // Sort by wordIndex (as integer)
    words.sort((a, b) => parseInt(a.dataset.wordIndex) - parseInt(b.dataset.wordIndex));
    const firstWord = words[0];
    const combinedText = words.map(w => w.textContent).join(' ');
    
    firstWord.textContent = combinedText;

    // Remove 'selected' class from the combined word
    firstWord.classList.remove('selected');

    // Remove other selected words
    words.slice(1).forEach(word => word.remove());
    selectedWords.clear();
});

// Separate Words
document.getElementById('separateBtn').addEventListener('click', () => {
    if (selectedWords.size !== 1) {
        alert('Please select exactly one word to separate');
        return;
    }
    
    const word = Array.from(selectedWords)[0];
    const text = word.textContent;
    const words = text.split(/\s+/).filter(Boolean);
    
    if (words.length <= 1) {
        alert('This word cannot be separated further');
        return;
    }
    
    const container = word.parentElement;
    let insertBeforeNode = word;
    words.forEach((w, i) => {
        const newWord = document.createElement('span');
        newWord.className = 'word';
        newWord.textContent = w;
        newWord.dataset.rowIndex = word.dataset.rowIndex;
        // Assign correct wordIndex based on position in container
        // Find the index where to insert
        let newIndex = Array.from(container.children).indexOf(word) + i;
        newWord.dataset.wordIndex = newIndex;
        newWord.addEventListener('click', () => {
            newWord.classList.toggle('selected');
            if (newWord.classList.contains('selected')) {
                selectedWords.add(newWord);
            } else {
                selectedWords.delete(newWord);
            }
        });
        container.insertBefore(newWord, insertBeforeNode);
        insertBeforeNode = newWord.nextSibling;
    });
    word.remove();
    selectedWords.clear();
});

// Mark as Aspect
document.getElementById('markAspectBtn').addEventListener('click', async () => {
    if (selectedWords.size === 0) {
        alert('Please select words to mark as aspect');
        return;
    }

    const words = Array.from(selectedWords);
    const aspects = words.map(w => w.textContent);
    
    try {
        const response = await fetch('/update_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                row_index: currentRowIndex,
                aspects: aspects
            })
        });

        if (response.ok) {
            words.forEach(word => {
                word.classList.add('aspect');
            });
            selectedWords.clear();
        }
    } catch (error) {
        console.error('Error updating aspects:', error);
    }
});

// Apply Sentiment
document.getElementById('applySentimentBtn').addEventListener('click', async () => {
    const sentiment = document.getElementById('sentimentSelect').value;
    if (!sentiment) {
        alert('Please select a sentiment');
        return;
    }

    if (selectedWords.size === 0) {
        alert('Please select words to apply sentiment');
        return;
    }

    const words = Array.from(selectedWords);
    const sentiments = {};
    words.forEach(word => {
        sentiments[word.dataset.wordIndex] = sentiment;
    });
    
    try {
        const response = await fetch('/update_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                row_index: currentRowIndex,
                sentiments: sentiments
            })
        });

        if (response.ok) {
            words.forEach(word => {
                word.classList.remove('sentiment-positive', 'sentiment-negative', 'sentiment-neutral');
                word.classList.add(`sentiment-${sentiment}`);
            });
            selectedWords.clear();
        }
    } catch (error) {
        console.error('Error updating sentiments:', error);
    }
});

// Save Annotations
document.getElementById('saveBtn').addEventListener('click', async () => {
    try {
        const response = await fetch('/save_data', {
            method: 'POST'
        });
        
        if (response.ok) {
            alert('Annotations saved successfully!');
        } else {
            alert('Error saving annotations');
        }
    } catch (error) {
        console.error('Error saving data:', error);
        alert('Error saving annotations');
    }
}); 