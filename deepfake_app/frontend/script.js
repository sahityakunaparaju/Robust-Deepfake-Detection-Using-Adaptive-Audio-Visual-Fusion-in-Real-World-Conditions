const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadSection = document.getElementById('uploadSection');
const processingOverlay = document.getElementById('processingOverlay');
const dashboardSection = document.getElementById('dashboardSection');
const progressFill = document.getElementById('progressFill');

const previewContainer = document.getElementById('previewContainer');
const videoPreview = document.getElementById('videoPreview');
const cancelUploadBtn = document.getElementById('cancelUploadBtn');
const confirmUploadBtn = document.getElementById('confirmUploadBtn');

const errorCard = document.getElementById('errorCard');
const errorTxt = document.getElementById('errorTxt');
const closeErrorBtn = document.getElementById('closeErrorBtn');

const apiUrl = 'http://localhost:8000/api/analyze';
const reportUrl = 'http://localhost:8000/api/report';

let currentFile = null;
let currentData = null;
let timelineChartInstance = null;

dropZone.addEventListener('click', () => fileInput.click());

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});
function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

dropZone.addEventListener('drop', handleDrop, false);
fileInput.addEventListener('change', handleFileSelect, false);

function handleDrop(e) {
    const file = e.dataTransfer.files[0];
    if (file) previewFile(file);
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) previewFile(file);
}

function previewFile(file) {
    if (!file.name.toLowerCase().match(/\.(mp4|avi)$/)) {
        showError('Please upload an MP4 or AVI file.');
        return;
    }
    errorCard.classList.add('hidden');
    currentFile = file;
    dropZone.classList.add('hidden');
    previewContainer.classList.remove('hidden');

    const fileURL = URL.createObjectURL(file);
    videoPreview.src = fileURL;
}

closeErrorBtn.addEventListener('click', () => {
    errorCard.classList.add('hidden');
});

cancelUploadBtn.addEventListener('click', resetUploadScreen);

confirmUploadBtn.addEventListener('click', () => {
    if (currentFile) startUpload(currentFile);
});

function showError(msg) {
    errorTxt.textContent = msg;
    errorCard.classList.remove('hidden');
}

function startUpload(file) {
    previewContainer.classList.add('hidden');
    processingOverlay.style.display = 'flex';

    let progress = 0;
    const simInterval = setInterval(() => {
        if (progress < 90) {
            progress += 5;
            progressFill.style.width = progress + '%';
        }
    }, 500);

    const formData = new FormData();
    formData.append('video', file);

    fetch(apiUrl, { method: 'POST', body: formData })
        .then(response => response.json())
        .then(data => {
            clearInterval(simInterval);
            processingOverlay.style.display = 'none';

            if (data.error) {
                resetUploadScreen();
                showError(data.error);
                return;
            }

            progressFill.style.width = '100%';
            currentData = data;
            setTimeout(() => showDashboard(data), 500);
        })
        .catch(error => {
            clearInterval(simInterval);
            processingOverlay.style.display = 'none';
            resetUploadScreen();
            showError(error.message || 'Error occurred during processing');
        });
}

function resetUploadScreen() {
    currentFile = null;
    videoPreview.src = "";
    previewContainer.classList.add('hidden');
    dropZone.classList.remove('hidden');
    fileInput.value = '';
    progressFill.style.width = '0%';
}

function showDashboard(data) {
    uploadSection.classList.add('hidden');
    dashboardSection.classList.remove('hidden');

    const predVal = document.getElementById('predictionValue');
    predVal.textContent = data.prediction;
    predVal.className = 'prediction-value ' + (data.prediction.toLowerCase() === 'fake' ? 'pred-fake' : 'pred-real');

    const confCircle = document.getElementById('confidenceCircle');
    const confVal = Math.round(data.confidence * 100);
    setTimeout(() => { confCircle.setAttribute('stroke-dasharray', `${confVal}, 100`); }, 100);
    confCircle.className.baseVal = 'circle ' + (data.prediction.toLowerCase() === 'fake' ? 'circ-fake' : 'circ-real');
    document.getElementById('confidenceText').textContent = confVal + '%';

    document.getElementById('decisionSummaryTxt').textContent = data.decision_summary || "Interpretation logic unavailable.";

    updateBar('videoScoreBar', 'videoScoreTxt', data.video_score || 0);
    updateBar('audioScoreBar', 'audioScoreTxt', data.audio_score || 0);
    updateBar('fakeRatioBar', 'fakeRatioTxt', data.fake_frame_ratio || 0);

    const m = data.metadata || {};
    document.getElementById('framesAnalyzedTxt').textContent = `${data.num_frames_analyzed || 0} / Total`;
    document.getElementById('stabilityTxt').textContent = (data.stability_score || 0).toFixed(2);
    document.getElementById('audioPresenceTxt').textContent = m.audio_detected ? 'Yes' : 'No';
    document.getElementById('procTimeTxt').textContent = `${m.processing_efficiency_fps || 0} fps`;

    document.getElementById('fwVideoTxt').textContent = data.fusion_weights?.video.toFixed(2) || '0.00';
    document.getElementById('fwAudioTxt').textContent = data.fusion_weights?.audio.toFixed(2) || '0.00';
    setTimeout(() => {
        document.getElementById('fwVideoBar').style.width = `${(data.fusion_weights?.video || 0) * 100}%`;
        document.getElementById('fwAudioBar').style.width = `${(data.fusion_weights?.audio || 0) * 100}%`;
    }, 300);

    renderTimelineChart(data.frame_predictions || []);
    renderTopFrames(data.top_frames || []);
}

function updateBar(barId, txtId, val) {
    const pc = Math.round(val * 100);
    document.getElementById(txtId).textContent = val.toFixed(2);
    setTimeout(() => { document.getElementById(barId).style.width = pc + '%'; }, 300);
}

function renderTimelineChart(predictions) {
    const ctx = document.getElementById('timelineChart').getContext('2d');
    if (timelineChartInstance) timelineChartInstance.destroy();

    const labels = predictions.map((_, i) => `Frame ${i + 1}`);
    const dataColors = predictions.map(p => p > 0.5 ? '#e63946' : '#2a9d8f');

    const gradient = ctx.createLinearGradient(0, 0, 0, 200);
    gradient.addColorStop(0, 'rgba(230, 57, 70, 0.5)');
    gradient.addColorStop(1, 'rgba(42, 157, 143, 0.1)');

    timelineChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Fake Probability',
                data: predictions,
                borderColor: '#d4a373',
                backgroundColor: gradient,
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointBackgroundColor: dataColors,
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { theme: 'dark' }
            },
            scales: {
                y: { min: 0, max: 1, grid: { color: '#333' }, ticks: { color: '#a0a0a0' } },
                x: { grid: { display: false }, ticks: { color: '#a0a0a0', maxTicksLimit: 10 } }
            }
        }
    });
}

function renderTopFrames(frames) {
    const container = document.getElementById('topFramesGrid');
    container.innerHTML = '';

    frames.forEach(frame => {
        const el = document.createElement('div');
        el.className = 'frame-card';

        el.innerHTML = `
            <img src="${frame.image_path}" />
            <div class="frame-info">
                <span>#${frame.frame_index}</span>
                <span>${frame.score.toFixed(2)}</span>
            </div>
        `;

        el.onclick = () => {
            window.currentFrameIndex = frame.frame_index;
            alert("Jumped to frame " + frame.frame_index); // you can later sync with chart
        };

        container.appendChild(el);
    });
}

document.getElementById('resetBtn').addEventListener('click', () => {
    dashboardSection.classList.add('hidden');
    uploadSection.classList.remove('hidden');
    errorCard.classList.add('hidden');
    resetUploadScreen();
});

document.getElementById('reportBtn').addEventListener('click', () => {
    if (!currentData) return;

    const originalText = document.getElementById('reportBtn').textContent;
    document.getElementById('reportBtn').textContent = "Generating...";
    document.getElementById('reportBtn').disabled = true;

    fetch(reportUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(currentData)
    })
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = 'deepfake_analysis_report.pdf';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);

            document.getElementById('reportBtn').textContent = originalText;
            document.getElementById('reportBtn').disabled = false;
        })
        .catch(err => {
            alert("Failed to generate PDF report.");
            document.getElementById('reportBtn').textContent = originalText;
            document.getElementById('reportBtn').disabled = false;
        });
});
