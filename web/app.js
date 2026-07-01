// DOM Elements
const elModelStatusDot = document.getElementById('model-status-dot');
const elModelStatusText = document.getElementById('model-status-text');
const elErrorAlert = document.getElementById('error-alert');
const elErrorMessage = document.getElementById('error-message');
const elLiveIndicator = document.getElementById('live-indicator');
const elInputModeText = document.getElementById('input-mode-text');
const elWebcam = document.getElementById('webcam');
const elCanvasOverlay = document.getElementById('canvas-overlay');
const elCanvasFace = document.getElementById('canvas-face');
const elImagePreview = document.getElementById('image-preview');
const elPlaceholder = document.getElementById('placeholder');
const elPredictionBadge = document.getElementById('prediction-badge');
const elBtnWebcam = document.getElementById('btn-webcam');
const elBtnWebcamText = document.getElementById('btn-webcam-text');
const elImageUpload = document.getElementById('image-upload');
const elBtnClear = document.getElementById('btn-clear');
const elResultIdle = document.getElementById('result-idle');
const elResultActive = document.getElementById('result-active');
const elResultLabel = document.getElementById('result-label');
const elResultConfidence = document.getElementById('result-confidence');
const elResultBar = document.getElementById('result-bar');
const elInferenceTime = document.getElementById('inference-time');

// State variables
let model = null;
let blazeModel = null;
let webcamStream = null;
let isWebcamRunning = false;
let animationFrameId = null;

// Load models
async function initModel() {
    try {
        showModelStatus('Memuat...', 'bg-amber-500 animate-pulse');
        
        // 1. Load Face Mask Classifier Model
        model = await tf.loadLayersModel('model/model.json');
        
        // 2. Load BlazeFace Detector Model
        blazeModel = await blazeface.load();
        
        // Warm up the models
        tf.tidy(() => {
            const dummy = tf.zeros([1, 224, 224, 3]);
            model.predict(dummy);
        });

        showModelStatus('Model Siap', 'bg-green-500');
        console.log("All models loaded successfully.");
    } catch (err) {
        showModelStatus('Gagal Memuat', 'bg-red-500');
        showError('Gagal memuat model. Pastikan file model.json dan group1-shard1of1.bin berada di folder web/model/ dan disajikan melalui server HTTP.', err);
    }
}

function showModelStatus(text, dotClass) {
    elModelStatusText.textContent = text;
    elModelStatusDot.className = `w-2 h-2 mr-1.5 rounded-full ${dotClass}`;
}

function showError(message, errorDetails) {
    elErrorAlert.classList.remove('hidden');
    elErrorMessage.innerHTML = `<strong>${message}</strong>${errorDetails ? `<br><span class="code-font text-xs opacity-80">${errorDetails.message || errorDetails}</span>` : ''}`;
    console.error(message, errorDetails);
}

function clearError() {
    elErrorAlert.classList.add('hidden');
}

// Preprocess visual element into standard input tensor [1, 224, 224, 3]
function preprocess(imageOrVideo) {
    return tf.tidy(() => {
        // Load pixel values as tensor (TF.js reads as RGB)
        let rgbTensor = tf.browser.fromPixels(imageOrVideo);
        
        // Swap channels from RGB to BGR because the OpenCV model in the notebook is trained on BGR images!
        let [r, g, b] = tf.split(rgbTensor, 3, 2);
        let bgrTensor = tf.concat([b, g, r], 2);
        
        // Resize to 224x224 (matching the model input shape)
        const resized = tf.image.resizeBilinear(bgrTensor, [224, 224]);
        
        // Normalization based on python preprocessing: face.astype("float32") / 255.0 (Scale [0, 1])
        const normalized = resized.div(255.0);
        
        // Add batch dimension [1, 224, 224, 3]
        return normalized.expandDims(0);
    });
}

// Run single prediction step with face detection
async function runInference(sourceElement) {
    if (!model || !blazeModel) return;
    
    const startTime = performance.now();
    
    // Get actual width and height of the visual source
    const w = sourceElement.videoWidth || sourceElement.naturalWidth || sourceElement.width || 0;
    const h = sourceElement.videoHeight || sourceElement.naturalHeight || sourceElement.height || 0;
    
    if (w === 0 || h === 0) return;
    
    // Setup and clear overlay canvas to match coordinates
    if (elCanvasOverlay.width !== w || elCanvasOverlay.height !== h) {
        elCanvasOverlay.width = w;
        elCanvasOverlay.height = h;
    }
    
    const ctxOverlay = elCanvasOverlay.getContext('2d');
    ctxOverlay.clearRect(0, 0, w, h);
    
    try {
        // 1. Run face detection
        const predictions = await blazeModel.estimateFaces(sourceElement, false);
        
        if (predictions.length > 0) {
            // Find face bounding box
            const pred = predictions[0];
            const start = pred.topLeft;
            const end = pred.bottomRight;
            
            const x = Math.max(0, start[0]);
            const y = Math.max(0, start[1]);
            const faceW = Math.min(w - x, end[0] - start[0]);
            const faceH = Math.min(h - y, end[1] - start[1]);
            
            // 2. Crop the face region and draw into hidden 224x224 canvas
            const ctxFace = elCanvasFace.getContext('2d');
            ctxFace.clearRect(0, 0, 224, 224);
            ctxFace.drawImage(sourceElement, x, y, faceW, faceH, 0, 0, 224, 224);
            
            // 3. Preprocess face crop and run classifier prediction
            const inputTensor = preprocess(elCanvasFace);
            const prediction = model.predict(inputTensor);
            const predictionData = await prediction.data();
            const value = predictionData[0]; // Sigmoid output [0, 1]
            
            // Clean up tensors
            inputTensor.dispose();
            prediction.dispose();
            
            const endTime = performance.now();
            const inferenceTimeMs = Math.round(endTime - startTime);
            
            // 4. Update classification UI
            const isMask = displayPrediction(value, inferenceTimeMs);
            
            // 5. Draw bounding box on overlay canvas
            ctxOverlay.strokeStyle = isMask ? '#22c55e' : '#ef4444'; // Green for Mask, Red for No Mask
            ctxOverlay.lineWidth = Math.max(3, Math.round(w / 150)); // Scale line width with frame size
            ctxOverlay.strokeRect(x, y, faceW, faceH);
            
            // Optional: Draw text label on top of bounding box
            ctxOverlay.fillStyle = isMask ? '#22c55e' : '#ef4444';
            const fontSize = Math.max(12, Math.round(w / 40));
            ctxOverlay.font = `bold ${fontSize}px sans-serif`;
            ctxOverlay.fillText(isMask ? "MASKER" : "TANPA MASKER", x, y > fontSize ? y - 6 : y + faceH + fontSize + 2);
            
        } else {
            // No faces detected
            ctxOverlay.clearRect(0, 0, w, h);
            
            elResultIdle.innerHTML = `
                <div class="flex flex-col items-center justify-center space-y-2">
                    <svg class="h-8 w-8 text-amber-500 animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                    <span>Wajah tidak terdeteksi. Posisikan wajah Anda di depan kamera.</span>
                </div>
            `;
            elResultIdle.classList.remove('hidden');
            elResultActive.classList.add('hidden');
            elPredictionBadge.classList.add('hidden');
        }
    } catch (err) {
        showError("Gagal menjalankan inferensi model.", err);
    }
}

function displayPrediction(val, timeMs) {
    // Notebook model decision logic:
    // val >= 0.5 => With Mask (Masker Terdeteksi)
    // val < 0.5 => Without Mask (Tanpa Masker)
    const isMask = val >= 0.5;
    
    // Confidence calculation
    let confidence = 0;
    if (isMask) {
        // Confidence for Mask
        confidence = val * 100;
    } else {
        // Confidence for No Mask
        confidence = (1.0 - val) * 100;
    }
    confidence = Math.max(0, Math.min(100, confidence));
    
    // Update UI elements
    elResultIdle.classList.add('hidden');
    elResultActive.classList.remove('hidden');
    
    const labelText = isMask ? "Masker Terdeteksi" : "Tanpa Masker";
    elResultLabel.textContent = labelText;
    elResultConfidence.textContent = `${confidence.toFixed(1)}%`;
    elInferenceTime.textContent = `Inferensi + Deteksi: ${timeMs} ms`;
    
    // Update progress bar width and color
    elResultBar.style.width = `${confidence}%`;
    
    if (isMask) {
        elResultLabel.className = "text-2xl font-bold tracking-tight text-green-500";
        elResultBar.className = "h-full bg-green-500 transition-all duration-200";
        
        elPredictionBadge.textContent = "MASKER";
        elPredictionBadge.className = "absolute top-4 right-4 px-3 py-1.5 rounded-lg text-xs font-bold shadow-lg border bg-green-950/80 border-green-700 text-green-400";
        elPredictionBadge.classList.remove('hidden');
    } else {
        elResultLabel.className = "text-2xl font-bold tracking-tight text-red-500";
        elResultBar.className = "h-full bg-red-500 transition-all duration-200";
        
        elPredictionBadge.textContent = "TANPA MASKER";
        elPredictionBadge.className = "absolute top-4 right-4 px-3 py-1.5 rounded-lg text-xs font-bold shadow-lg border bg-red-950/80 border-red-700 text-red-400";
        elPredictionBadge.classList.remove('hidden');
    }
    
    return isMask;
}

// Live prediction loop for webcam
async function webcamLoop() {
    if (!isWebcamRunning) return;
    
    await runInference(elWebcam);
    
    // Schedule next frame
    animationFrameId = requestAnimationFrame(webcamLoop);
}

// Webcam start/stop handler
async function toggleWebcam() {
    clearError();
    if (isWebcamRunning) {
        stopWebcam();
    } else {
        await startWebcam();
    }
}

async function startWebcam() {
    if (!model || !blazeModel) {
        showError("Model belum siap. Tunggu hingga model selesai dimuat.");
        return;
    }
    
    try {
        elBtnWebcamText.textContent = "Meminta Akses...";
        elBtnWebcam.disabled = true;
        
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            },
            audio: false
        });
        
        elWebcam.srcObject = webcamStream;
        elWebcam.classList.remove('hidden');
        elCanvasOverlay.classList.remove('hidden');
        elPlaceholder.classList.add('hidden');
        elImagePreview.classList.add('hidden');
        elBtnClear.classList.remove('hidden');
        
        // Wait for video metadata to load so size dimensions are valid
        elWebcam.onloadedmetadata = () => {
            elWebcam.play();
            isWebcamRunning = true;
            elBtnWebcam.disabled = false;
            elBtnWebcamText.textContent = "Hentikan Kamera";
            elBtnWebcam.className = "px-4 py-2.5 rounded-xl bg-red-600 hover:bg-red-500 text-slate-100 font-semibold text-sm transition flex items-center space-x-2 shadow-lg shadow-red-600/10";
            
            elLiveIndicator.className = "w-2 h-2 rounded-full bg-red-500 animate-ping";
            elInputModeText.textContent = "Kamera Aktif";
            
            // Set dimensions of overlay to match video feed
            elCanvasOverlay.width = elWebcam.videoWidth;
            elCanvasOverlay.height = elWebcam.videoHeight;
            
            // Start prediction loop
            webcamLoop();
        };
    } catch (err) {
        stopWebcam();
        showError("Akses kamera ditolak atau tidak tersedia.", err);
    }
}

function stopWebcam() {
    isWebcamRunning = false;
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    
    elWebcam.srcObject = null;
    elWebcam.classList.add('hidden');
    elCanvasOverlay.classList.add('hidden');
    
    elBtnWebcam.disabled = false;
    elBtnWebcamText.textContent = "Mulai Kamera";
    elBtnWebcam.className = "px-4 py-2.5 rounded-xl bg-green-600 hover:bg-green-500 text-slate-950 font-semibold text-sm transition flex items-center space-x-2 shadow-lg shadow-green-600/10";
    
    elLiveIndicator.className = "w-2 h-2 rounded-full bg-slate-600";
    elInputModeText.textContent = "Kamera Nonaktif";
}

// Static image file upload handler
elImageUpload.addEventListener('change', (e) => {
    clearError();
    const file = e.target.files[0];
    if (!file) return;
    
    stopWebcam();
    
    const reader = new FileReader();
    reader.onload = (event) => {
        elImagePreview.src = event.target.result;
        
        elImagePreview.onload = () => {
            elImagePreview.classList.remove('hidden');
            elCanvasOverlay.classList.remove('hidden');
            elPlaceholder.classList.add('hidden');
            elWebcam.classList.add('hidden');
            elBtnClear.classList.remove('hidden');
            
            elInputModeText.textContent = `File: ${file.name.substring(0, 15)}...`;
            elLiveIndicator.className = "w-2 h-2 rounded-full bg-blue-500";
            
            // Set dimensions of overlay to match image
            elCanvasOverlay.width = elImagePreview.naturalWidth || elImagePreview.width;
            elCanvasOverlay.height = elImagePreview.naturalHeight || elImagePreview.height;
            
            // Run inference on the uploaded image
            if (model && blazeModel) {
                runInference(elImagePreview);
            } else {
                showError("Model belum siap. Tunggu hingga model selesai dimuat.");
            }
        };
    };
    reader.readAsDataURL(file);
});

// Clear/Reset input handler
elBtnClear.addEventListener('click', () => {
    stopWebcam();
    
    elImagePreview.src = '';
    elImagePreview.classList.add('hidden');
    elWebcam.classList.add('hidden');
    elCanvasOverlay.classList.add('hidden');
    elPlaceholder.classList.remove('hidden');
    elBtnClear.classList.add('hidden');
    elPredictionBadge.classList.add('hidden');
    elImageUpload.value = '';
    
    elInputModeText.textContent = "Tidak ada input";
    elLiveIndicator.className = "w-2 h-2 rounded-full bg-slate-600";
    
    elResultIdle.innerHTML = "Menunggu input visual untuk melakukan inferensi...";
    elResultIdle.classList.remove('hidden');
    elResultActive.classList.add('hidden');
    clearError();
});

// Event listener for webcam button
elBtnWebcam.addEventListener('click', toggleWebcam);

// Initialize model on page load
window.addEventListener('DOMContentLoaded', initModel);
