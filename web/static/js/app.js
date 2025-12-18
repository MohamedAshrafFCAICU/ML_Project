let currentModel = 'svm';
let videoStream = null;
let isPaused = false;
let classifyInterval = null;
let facingMode = 'environment';
let uploadedImage = null;

const classColors = {
    'cardboard': '#8B4513',
    'glass': '#00FF00',
    'metal': '#C0C0C0',
    'paper': '#FFFF00',
    'plastic': '#0000FF',
    'trash': '#808080',
    'unknown': '#FF00FF'
};

document.addEventListener('DOMContentLoaded', () => {
    setupDragDrop();
    updateResultsPanel(null);
});

function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`tab-${tabName}`).classList.add('active');
    event.target.classList.add('active');
    if (tabName !== 'camera' && videoStream) stopCamera();
}

function selectModel(model) {
    currentModel = model;
    document.querySelectorAll('.model-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`btn-${model}`).classList.add('active');
}

async function startCamera() {
    try {
        const constraints = { video: { facingMode: facingMode, width: { ideal: 640 }, height: { ideal: 480 } } };
        videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        document.getElementById('video').srcObject = videoStream;
        document.getElementById('camera-overlay').classList.add('hidden');
        document.getElementById('btn-start').disabled = true;
        document.getElementById('btn-pause').disabled = false;
        document.getElementById('btn-stop').disabled = false;
        document.getElementById('btn-screenshot').disabled = false;
        isPaused = false;
        startClassification();
    } catch (err) {
        alert('Cannot access camera. Please allow camera permissions.');
    }
}

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    if (classifyInterval) {
        clearInterval(classifyInterval);
        classifyInterval = null;
    }
    document.getElementById('video').srcObject = null;
    document.getElementById('camera-overlay').classList.remove('hidden');
    document.getElementById('overlay-text').textContent = 'Click Start to begin';
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-pause').disabled = true;
    document.getElementById('btn-stop').disabled = true;
    document.getElementById('btn-screenshot').disabled = true;
    updateResultsPanel(null);
}

function togglePause() {
    isPaused = !isPaused;
    const btn = document.getElementById('btn-pause');
    if (isPaused) {
        btn.textContent = '▶ Resume';
        document.getElementById('overlay-text').textContent = 'PAUSED';
        document.getElementById('camera-overlay').classList.remove('hidden');
    } else {
        btn.textContent = '⏸ Pause';
        document.getElementById('camera-overlay').classList.add('hidden');
    }
}

async function switchCamera() {
    facingMode = facingMode === 'user' ? 'environment' : 'user';
    if (videoStream) {
        stopCamera();
        await startCamera();
    }
}

function takeScreenshot() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const link = document.createElement('a');
    link.download = `waste_screenshot_${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
}

function startClassification() {
    classifyInterval = setInterval(async () => {
        if (isPaused || !videoStream) return;
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        try {
            const response = await fetch('/api/classify_base64', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData, model: currentModel })
            });
            const result = await response.json();
            updateResultsPanel(result);
        } catch (err) {}
    }, 1000);
}

function setupDragDrop() {
    const dropArea = document.getElementById('drop-area');
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, e => { e.preventDefault(); e.stopPropagation(); }, false);
    });
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
    });
    dropArea.addEventListener('drop', e => {
        const files = e.dataTransfer.files;
        if (files.length > 0) handleImageFile(files[0]);
    }, false);
}

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file) handleImageFile(file);
}

function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }
    uploadedImage = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('image-preview').src = e.target.result;
        document.getElementById('image-preview-container').style.display = 'block';
    };
    reader.readAsDataURL(file);
}

async function classifyUploadedImage() {
    if (!uploadedImage) return;
    const formData = new FormData();
    formData.append('image', uploadedImage);
    formData.append('model', currentModel);
    try {
        const response = await fetch('/api/classify', { method: 'POST', body: formData });
        const result = await response.json();
        updateResultsPanel(result);
    } catch (err) {
        alert('Error classifying image');
    }
}

async function handleFolderUpload(event) {
    const files = Array.from(event.target.files).filter(f => f.type.startsWith('image/'));
    if (files.length === 0) {
        alert('No images found in folder');
        return;
    }
    document.getElementById('folder-results').style.display = 'block';
    document.getElementById('folder-count').textContent = files.length;
    const results = [];
    const progressFill = document.getElementById('progress-fill');
    for (let i = 0; i < files.length; i++) {
        const formData = new FormData();
        formData.append('image', files[i]);
        formData.append('model', currentModel);
        const thumbnail = await createThumbnail(files[i]);
        try {
            const response = await fetch('/api/classify', { method: 'POST', body: formData });
            const result = await response.json();
            results.push({ filename: files[i].name, thumbnail: thumbnail, class: result.class_name, confidence: result.confidence });
        } catch (err) {
            results.push({ filename: files[i].name, thumbnail: thumbnail, class: 'error', confidence: 0 });
        }
        progressFill.style.width = `${((i + 1) / files.length) * 100}%`;
    }
    displayFolderResults(results);
}

function createThumbnail(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                const size = 60;
                canvas.width = size;
                canvas.height = size;
                const ctx = canvas.getContext('2d');
                const minDim = Math.min(img.width, img.height);
                const sx = (img.width - minDim) / 2;
                const sy = (img.height - minDim) / 2;
                ctx.drawImage(img, sx, sy, minDim, minDim, 0, 0, size, size);
                resolve(canvas.toDataURL('image/jpeg', 0.7));
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    });
}

function displayFolderResults(results) {
    let html = `<table class="folder-table"><thead><tr><th>Image</th><th>Filename</th><th>Class</th><th>Confidence</th></tr></thead><tbody>`;
    results.forEach(r => {
        const color = classColors[r.class] || '#fff';
        html += `<tr><td><img src="${r.thumbnail}" class="folder-thumbnail" alt="${r.filename}"></td><td>${r.filename}</td><td style="color: ${color}">${r.class.toUpperCase()}</td><td>${(r.confidence * 100).toFixed(1)}%</td></tr>`;
    });
    html += '</tbody></table>';
    const summary = {};
    results.forEach(r => { summary[r.class] = (summary[r.class] || 0) + 1; });
    let summaryHtml = '<div style="margin-top: 20px;"><strong>Summary:</strong> ';
    for (const [cls, count] of Object.entries(summary)) {
        const color = classColors[cls] || '#fff';
        summaryHtml += `<span style="color: ${color}">${cls}: ${count}</span> | `;
    }
    summaryHtml = summaryHtml.slice(0, -3) + '</div>';
    document.getElementById('folder-table-container').innerHTML = html + summaryHtml;
}

function updateResultsPanel(result) {
    const classEl = document.getElementById('result-class');
    const confEl = document.getElementById('result-confidence');
    const barsEl = document.getElementById('result-bars');
    if (!result || result.error) {
        classEl.textContent = '-';
        classEl.style.color = '#888';
        confEl.textContent = 'Confidence: -';
        barsEl.innerHTML = '';
        return;
    }
    classEl.textContent = result.class_name.toUpperCase();
    classEl.style.color = classColors[result.class_name] || '#00ff88';
    confEl.textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
    let barsHtml = '';
    const probs = result.probabilities || {};
    const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    for (const [cls, prob] of sorted) {
        const color = classColors[cls] || '#888';
        const width = (prob * 100).toFixed(1);
        barsHtml += `<div class="prob-row"><div class="prob-label">${cls}</div><div class="prob-bar-container"><div class="prob-bar" style="width: ${width}%; background: ${color}"></div></div><div class="prob-value">${width}%</div></div>`;
    }
    barsEl.innerHTML = barsHtml;
}
