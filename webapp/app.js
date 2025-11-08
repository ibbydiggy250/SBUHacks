// MediaPipe Hand Tracking Integration for ASL Detection
// Enhanced version with better error handling and debugging

const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('canvas');
const canvasCtx = canvasElement.getContext('2d');
const statusText = document.getElementById('status');
const handCountText = document.getElementById('hand-count');
const handDetailsDiv = document.getElementById('hand-details');

// Feature storage (mimics the Python tracker's 84-dimensional output)
let currentFeatures = new Array(84).fill(0);
let leftHandFeatures = new Array(42).fill(0);
let rightHandFeatures = new Array(42).fill(0);

// Drawing styles (matching Python tracker colors)
const LEFT_HAND_COLOR = '#00FF00';  // Green for left hand
const RIGHT_HAND_COLOR = '#FF0000'; // Red for right hand

// Hand connections (MediaPipe hand topology)
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],           // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],           // Index finger
  [0, 9], [9, 10], [10, 11], [11, 12],      // Middle finger
  [0, 13], [13, 14], [14, 15], [15, 16],    // Ring finger
  [0, 17], [17, 18], [18, 19], [19, 20],    // Pinky
  [5, 9], [9, 13], [13, 17]                 // Palm
];

// Extract hand landmarks to feature vector
function extractHandFeatures(landmarks) {
  const features = [];
  for (const landmark of landmarks) {
    features.push(landmark.x);
    features.push(landmark.y);
  }
  return features;
}

// Draw landmarks on canvas
function drawLandmarks(ctx, landmarks, color) {
  ctx.fillStyle = color;
  ctx.strokeStyle = color;
  
  for (const landmark of landmarks) {
    const x = landmark.x * canvasElement.width;
    const y = landmark.y * canvasElement.height;
    
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fill();
  }
}

// Draw connections between landmarks
function drawConnections(ctx, landmarks, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;

  for (const connection of HAND_CONNECTIONS) {
    const start = landmarks[connection[0]];
    const end = landmarks[connection[1]];
    
    const startX = start.x * canvasElement.width;
    const startY = start.y * canvasElement.height;
    const endX = end.x * canvasElement.width;
    const endY = end.y * canvasElement.height;

    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.stroke();
  }
}

// Process results from MediaPipe
function onResults(results) {
  // Match canvas size to video
  canvasElement.width = videoElement.videoWidth;
  canvasElement.height = videoElement.videoHeight;

  // Clear canvas
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  // Reset features
  leftHandFeatures = new Array(42).fill(0);
  rightHandFeatures = new Array(42).fill(0);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    handCountText.textContent = results.multiHandLandmarks.length;
    handDetailsDiv.innerHTML = '';

    for (let i = 0; i < results.multiHandLandmarks.length; i++) {
      const landmarks = results.multiHandLandmarks[i];
      // Swap handedness because camera is mirrored
      const rawHandedness = results.multiHandedness[i].label;
      const handedness = rawHandedness === 'Left' ? 'Right' : 'Left';
      const score = results.multiHandedness[i].score;

      // Extract features
      const features = extractHandFeatures(landmarks);
      
      if (handedness === 'Left') {
        leftHandFeatures = features;
      } else {
        rightHandFeatures = features;
      }

      // Choose color
      const handColor = handedness === 'Left' ? LEFT_HAND_COLOR : RIGHT_HAND_COLOR;

      // Draw
      drawConnections(canvasCtx, landmarks, handColor);
      drawLandmarks(canvasCtx, landmarks, handColor);

      // Update info panel
      const handDiv = document.createElement('div');
      handDiv.className = 'hand-detail';
      handDiv.innerHTML = `
        <span class="hand-label" style="color: ${handColor}">
          ${handedness} Hand
        </span>
        <span class="confidence">${(score * 100).toFixed(1)}%</span>
      `;
      handDetailsDiv.appendChild(handDiv);
    }

    statusText.textContent = "✅ Tracking Active";
    statusText.style.color = "#00ff00";
  } else {
    handCountText.textContent = '0';
    handDetailsDiv.innerHTML = '<div class="no-hands">Show your hands to the camera</div>';
    statusText.textContent = "👋 Show your hands";
    statusText.style.color = "#ffaa00";
  }

  // Combine features
  currentFeatures = [...leftHandFeatures, ...rightHandFeatures];
}

// Initialize camera and MediaPipe
async function init() {
  try {
    statusText.textContent = "📷 Requesting camera access...";
    
    // Get camera stream
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    });
    
    videoElement.srcObject = stream;
    
    // Wait for video to load
    await new Promise((resolve) => {
      videoElement.onloadedmetadata = () => {
        videoElement.play();
        resolve();
      };
    });
    
    statusText.textContent = "⚙️ Initializing hand tracking...";
    
    // Initialize MediaPipe Hands
    const hands = new Hands({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
      }
    });

    hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    hands.onResults(onResults);

    // Create camera helper
    const camera = new Camera(videoElement, {
      onFrame: async () => {
        await hands.send({ image: videoElement });
      },
      width: 1280,
      height: 720
    });

    // Start camera
    await camera.start();
    
    statusText.textContent = "✅ Ready - Show your hands!";
    statusText.style.color = "#00ff00";
    
  } catch (err) {
    console.error('Error:', err);
    statusText.textContent = "❌ Error: " + err.message;
    statusText.style.color = "#ff0000";
  }
}

// Export features
function getCurrentFeatures() {
  return currentFeatures;
}

// Wait for MediaPipe libraries to load
if (typeof Hands !== 'undefined') {
  init();
} else {
  // Wait for libraries to load
  window.addEventListener('load', () => {
    setTimeout(init, 1000);
  });
}

window.getCurrentFeatures = getCurrentFeatures;
