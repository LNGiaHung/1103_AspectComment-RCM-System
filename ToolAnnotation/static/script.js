document.addEventListener('DOMContentLoaded', function() {
    const textContainer = document.getElementById('text-to-label');
    const roleSelect = document.getElementById('role-select');
    const saveButton = document.getElementById('save-label');
    const labelsList = document.getElementById('labels-list');
    const uploadForm = document.getElementById('upload-form');
    const csvFileInput = document.getElementById('csv-file');
    const exportButton = document.getElementById('export-csv');
    const prevTextButton = document.getElementById('prev-text');
    const nextTextButton = document.getElementById('next-text');
    const textCounter = document.getElementById('text-counter');
    
    let selectedWords = [];
    let labels = [];
    let texts = [];
    let currentTextIndex = 0;

    // Handle CSV upload
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const file = csvFileInput.files[0];
        if (!file) {
            alert('Please select a CSV file');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        fetch('/api/upload_csv', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }
            texts = data.texts;
            currentTextIndex = 0;
            updateTextDisplay();
            updateTextCounter();
        })
        .catch(error => console.error('Error:', error));
    });

    // Handle text navigation
    prevTextButton.addEventListener('click', function() {
        if (currentTextIndex > 0) {
            currentTextIndex--;
            updateTextDisplay();
            updateTextCounter();
        }
    });

    nextTextButton.addEventListener('click', function() {
        if (currentTextIndex < texts.length - 1) {
            currentTextIndex++;
            updateTextDisplay();
            updateTextCounter();
        }
    });

    function updateTextDisplay() {
        const text = texts[currentTextIndex] || '';
        // Split text into words and create spans for each word
        const words = text.split(/\s+/);
        textContainer.innerHTML = words.map((word, index) => 
            `<span class="word" data-word-index="${index}">${word}</span>`
        ).join(' ');
        
        // Add click event listeners to words
        document.querySelectorAll('.word').forEach(word => {
            word.addEventListener('click', handleWordClick);
        });

        // Overlay: highlight labeled words for this sentence
        applyLabelOverlays();
    }

    function handleWordClick(event) {
        const word = event.target;
        const wordIndex = parseInt(word.dataset.wordIndex);
        
        if (word.classList.contains('selected')) {
            // Deselect the word
            word.classList.remove('selected');
            selectedWords = selectedWords.filter(w => w.index !== wordIndex);
        } else {
            // Select the word
            word.classList.add('selected');
            selectedWords.push({
                index: wordIndex,
                text: word.textContent
            });
        }
        
        // Sort selected words by index
        selectedWords.sort((a, b) => a.index - b.index);
    }

    function applyLabelOverlays() {
        // Remove all label overlays first
        document.querySelectorAll('.word').forEach(word => {
            word.className = 'word';
        });
        // Get labels for current sentence
        const currentLabels = labels.filter(label => label.sentence_index === currentTextIndex);
        currentLabels.forEach(label => {
            label.word_indices.forEach(idx => {
                const wordSpan = document.querySelector(`.word[data-word-index='${idx}']`);
                if (wordSpan) {
                    wordSpan.classList.add('selected');
                    wordSpan.classList.add(label.role);
                }
            });
        });
    }

    function updateTextCounter() {
        textCounter.textContent = `Text ${currentTextIndex + 1} of ${texts.length}`;
    }

    // Save label
    saveButton.addEventListener('click', function() {
        if (selectedWords.length === 0) {
            alert('Please select some words first!');
            return;
        }

        const role = roleSelect.value;
        const selectedText = selectedWords.map(w => w.text).join(' ');
        
        const label = {
            text: selectedText,
            role: role,
            id: Date.now(),
            sentence: texts[currentTextIndex],
            sentence_index: currentTextIndex,
            word_indices: selectedWords.map(w => w.index)
        };

        // Add to local array
        labels.push(label);

        // Save to server
        fetch('/api/save_label', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(label)
        })
        .then(response => response.json())
        .then(data => {
            updateLabelsList();
            // Clear selection
            selectedWords = [];
            document.querySelectorAll('.word.selected').forEach(word => {
                word.classList.remove('selected');
            });
            // Re-apply overlays
            applyLabelOverlays();
        })
        .catch(error => console.error('Error:', error));
    });

    // Export labels
    exportButton.addEventListener('click', function() {
        if (labels.length === 0) {
            alert('No labels to export!');
            return;
        }

        window.location.href = '/api/export_csv';
    });

    // Update labels list
    function updateLabelsList() {
        labelsList.innerHTML = '';
        // Filter labels for current text
        const currentLabels = labels.filter(label => label.sentence_index === currentTextIndex);
        
        currentLabels.forEach(label => {
            const labelItem = document.createElement('div');
            labelItem.className = 'label-item';
            labelItem.innerHTML = `
                <span class="remove-label" data-id="${label.id}">×</span>
                <strong>${label.role}:</strong> ${label.text}
            `;
            labelsList.appendChild(labelItem);
        });

        // Add remove functionality
        document.querySelectorAll('.remove-label').forEach(button => {
            button.addEventListener('click', function() {
                const id = parseInt(this.dataset.id);
                labels = labels.filter(label => label.id !== id);
                updateLabelsList();
            });
        });
    }

    // Load existing labels
    fetch('/api/get_labels')
        .then(response => response.json())
        .then(data => {
            labels = data;
            updateLabelsList();
        })
        .catch(error => console.error('Error:', error));
}); 