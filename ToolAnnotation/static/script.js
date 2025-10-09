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

    // --- BEGIN NEW STATE AND UTILS ---
    const wordsContainer = document.getElementById('words-container');
    const combineBtn = document.getElementById('combine-words');
    const separateBtn = document.getElementById('separate-word');
    const resetBtn = document.getElementById('reset-labels');

    let wordGroups = []; // [{text: '...', indices: [0,1], label: null}]
    let selectedGroupIndices = []; // indices in wordGroups
    let allLabels = []; // [{sentence_index, groups: [{text, indices, label}]}]

    function splitTextToGroups(text) {
        const words = text.split(/\s+/);
        return words.map((w, i) => ({ text: w, indices: [i], label: null }));
    }

    function renderWordGroups() {
        wordsContainer.innerHTML = '';
        wordGroups.forEach((group, idx) => {
            const span = document.createElement('span');
            span.className = 'word-group btn btn-light btn-sm m-1';
            span.textContent = group.text;
            span.dataset.groupIndex = idx;
            if (selectedGroupIndices.includes(idx)) span.classList.add('selected');
            if (group.label) span.classList.add('labeled');
            span.onclick = () => toggleGroupSelect(idx);
            wordsContainer.appendChild(span);
        });
        updateNextButtonState();
    }

    function toggleGroupSelect(idx) {
        if (selectedGroupIndices.includes(idx)) {
            selectedGroupIndices = selectedGroupIndices.filter(i => i !== idx);
        } else {
            selectedGroupIndices.push(idx);
        }
        renderWordGroups();
    }

    function combineSelectedGroups() {
        if (selectedGroupIndices.length < 2) return;
        selectedGroupIndices.sort((a, b) => a - b);
        const combined = {
            text: selectedGroupIndices.map(i => wordGroups[i].text).join(' '),
            indices: selectedGroupIndices.flatMap(i => wordGroups[i].indices),
            label: null
        };
        wordGroups = wordGroups.filter((_, i) => !selectedGroupIndices.includes(i));
        wordGroups.splice(selectedGroupIndices[0], 0, combined);
        selectedGroupIndices = [selectedGroupIndices[0]];
        renderWordGroups();
    }

    function separateSelectedGroup() {
        if (selectedGroupIndices.length !== 1) return;
        const idx = selectedGroupIndices[0];
        const group = wordGroups[idx];
        if (group.indices.length === 1) return;
        const newGroups = group.indices.map((i, j) => ({
            text: group.text.split(' ')[j],
            indices: [i],
            label: null
        }));
        wordGroups.splice(idx, 1, ...newGroups);
        selectedGroupIndices = newGroups.map((_, j) => idx + j);
        renderWordGroups();
    }

    function resetLabels() {
        wordGroups.forEach(g => g.label = null);
        selectedGroupIndices = [];
        renderWordGroups();
        updateLabelsList();
    }

    function labelSelectedGroup() {
        if (selectedGroupIndices.length !== 1) {
            alert('Select one word/group to label.');
            return;
        }
        const idx = selectedGroupIndices[0];
        const role = roleSelect.value;
        wordGroups[idx].label = role;
        selectedGroupIndices = [];
        renderWordGroups();
        updateLabelsList();
    }

    function updateLabelsList() {
        labelsList.innerHTML = '';
        wordGroups.forEach((g, idx) => {
            const div = document.createElement('div');
            div.className = 'label-item';
            div.innerHTML = `<strong>${g.label || '-'}:</strong> ${g.text}`;
            if (g.label) {
                const removeBtn = document.createElement('span');
                removeBtn.textContent = '\u00d7';
                removeBtn.className = 'remove-label';
                removeBtn.onclick = () => { g.label = null; renderWordGroups(); updateLabelsList(); };
                div.prepend(removeBtn);
            }
            labelsList.appendChild(div);
        });
    }

    function updateNextButtonState() {
        const allLabeled = wordGroups.every(g => g.label);
        nextTextButton.disabled = !allLabeled;
    }

    function saveCurrentLabelsToAllLabels() {
        allLabels = allLabels.filter(l => l.sentence_index !== currentTextIndex);
        allLabels.push({
            sentence_index: currentTextIndex,
            groups: wordGroups.map(g => ({ text: g.text, indices: g.indices, label: g.label }))
        });
    }

    function exportLabelsToCSV() {
        // Prepare CSV content
        let csv = 'sentence_index,group_text,indices,label\n';
        allLabels.forEach(l => {
            l.groups.forEach(g => {
                csv += `${l.sentence_index},"${g.text}","${g.indices.join(' ')}",${g.label}\n`;
            });
        });
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'labels.csv';
        a.click();
        URL.revokeObjectURL(url);
    }
    // --- END NEW STATE AND UTILS ---

    // --- OVERRIDE NAVIGATION AND LABELING LOGIC ---
    function updateTextDisplay() {
        const text = texts[currentTextIndex] || '';
        wordGroups = splitTextToGroups(text);
        selectedGroupIndices = [];
        // Restore previous labels if any
        const prev = allLabels.find(l => l.sentence_index === currentTextIndex);
        if (prev) {
            wordGroups.forEach((g, i) => {
                const found = prev.groups.find(pg => pg.indices.join(',') === g.indices.join(','));
                if (found) g.label = found.label;
            });
        }
        renderWordGroups();
        updateLabelsList();
    }

    prevTextButton.onclick = function() {
        if (currentTextIndex > 0) {
            saveCurrentLabelsToAllLabels();
            currentTextIndex--;
            updateTextDisplay();
            updateTextCounter();
        }
    };

    nextTextButton.onclick = function() {
        if (currentTextIndex < texts.length - 1 && wordGroups.every(g => g.label)) {
            saveCurrentLabelsToAllLabels();
            currentTextIndex++;
            updateTextDisplay();
            updateTextCounter();
        }
    };

    combineBtn.onclick = combineSelectedGroups;
    separateBtn.onclick = separateSelectedGroup;
    resetBtn.onclick = resetLabels;
    saveButton.onclick = labelSelectedGroup;
    exportButton.onclick = exportLabelsToCSV;

    // Load existing labels
    fetch('/api/get_labels')
        .then(response => response.json())
        .then(data => {
            allLabels = data;
            // Update wordGroups and render based on existing labels
            allLabels.forEach(l => {
                l.groups.forEach(g => {
                    wordGroups.push({ text: g.text, indices: g.indices, label: g.label });
                });
            });
            renderWordGroups();
            updateLabelsList();
        })
        .catch(error => console.error('Error:', error));
}); 