document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const canvas = document.getElementById('input-canvas');
    const ctx = canvas.getContext('2d');
    const uploadBox = document.getElementById('upload-box');
    const fileInput = document.getElementById('file-input');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resetBtn = document.getElementById('reset-btn');
    const generateBtn = document.getElementById('generate-btn');
    const textPromptContainer = document.getElementById('text-prompt-container');
    const textInput = document.getElementById('text-input');
    const textSubmitBtn = document.getElementById('text-submit-btn');
    const promptBtns = document.querySelectorAll('.prompt-btn');
    const instructionSets = document.querySelectorAll('.instruction-set');
    const sampleModal = document.getElementById('sample-modal');
    const sampleBtn = document.getElementById('sample-btn');
    const closeModal = document.querySelector('.close-modal');
    const sampleImages = document.querySelectorAll('.sample-image');

    // State
    let currentImage = null;
    let promptType = 'point';
    let scribblePoints = [];
    let boxPoints = [];
    let isDrawing = false;
    let canvasBounds = null;

    // Initialize
    function init() {
        // Set up event listeners
        setupEventListeners();

        // Create sample directory
        createSampleDirectory();
    }

    // Create sample directory if it doesn't exist
    function createSampleDirectory() {
        fetch('/api/create_samples', {
            method: 'POST',
        })
        .then(response => response.json())
        .catch(error => console.error('Error creating sample directory:', error));
    }

    // Set up event listeners
    function setupEventListeners() {
        // Prompt type buttons
        promptBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                setPromptType(btn.id.split('-')[0]);
            });
        });

        // Upload box events
        uploadBox.addEventListener('click', () => {
            fileInput.click();
        });

        uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadBox.classList.add('drag-over');
        });

        uploadBox.addEventListener('dragleave', () => {
            uploadBox.classList.remove('drag-over');
        });

        uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadBox.classList.remove('drag-over');

            if (e.dataTransfer.files.length) {
                handleFileUpload(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) {
                handleFileUpload(fileInput.files[0]);
            }
        });

        // Canvas interaction events
        canvas.addEventListener('click', handleCanvasClick);
        canvas.addEventListener('mousedown', (e) => {
            if (promptType === 'scribble') {
                isDrawing = true;
                const point = getCanvasPoint(e);
                addScribblePoint(point);
            }
        });

        canvas.addEventListener('mousemove', (e) => {
            if (isDrawing && promptType === 'scribble') {
                const point = getCanvasPoint(e);
                addScribblePoint(point);
            }
        });

        canvas.addEventListener('mouseup', () => {
            isDrawing = false;
        });

        canvas.addEventListener('mouseleave', () => {
            isDrawing = false;
        });

        // Button events
        resetBtn.addEventListener('click', resetCanvas);
        generateBtn.addEventListener('click', generateSegmentation);
        textSubmitBtn.addEventListener('click', handleTextSubmit);

        // Sample image modal
        sampleBtn.addEventListener('click', () => {
            sampleModal.style.display = 'block';
        });

        closeModal.addEventListener('click', () => {
            sampleModal.style.display = 'none';
        });

        window.addEventListener('click', (e) => {
            if (e.target === sampleModal) {
                sampleModal.style.display = 'none';
            }
        });

        sampleImages.forEach(img => {
            img.addEventListener('click', () => {
                const imgSrc = img.getAttribute('data-src');
                loadSampleImage(imgSrc);
                sampleModal.style.display = 'none';
            });
        });
    }

    // Set the prompt type
    function setPromptType(type) {
        promptType = type;

        // Update UI
        promptBtns.forEach(btn => {
            if (btn.id === `${type}-prompt`) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Show/hide text prompt input
        textPromptContainer.style.display = type === 'text' ? 'block' : 'none';

        // Show appropriate instructions
        instructionSets.forEach(set => {
            set.style.display = 'none';
        });
        document.getElementById(`${type}-instructions`).style.display = 'block';

        // Reset points for drawing
        scribblePoints = [];
        boxPoints = [];

        // Redraw canvas
        if (currentImage) {
            drawImageOnCanvas(currentImage);
        }

        // Update generate button visibility
        updateGenerateButtonState();
    }

    // Update generate button state based on current state
    function updateGenerateButtonState() {
        if (!currentImage) {
            generateBtn.disabled = true;
            return;
        }

        if (promptType === 'scribble') {
            generateBtn.disabled = scribblePoints.length < 2;
        } else if (promptType === 'box') {
            generateBtn.disabled = boxPoints.length < 2;
        } else if (promptType === 'text') {
            generateBtn.disabled = true; // Text has its own submit button
        } else {
            generateBtn.disabled = false;
        }
    }

    // Handle file upload
    function handleFileUpload(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                currentImage = img;
                drawImageOnCanvas(img);
                resetInteractionState();
                updateGenerateButtonState();
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    // Load a sample image
    function loadSampleImage(src) {
        const img = new Image();
        img.onload = () => {
            currentImage = img;
            drawImageOnCanvas(img);
            resetInteractionState();
            updateGenerateButtonState();
        };
        img.onerror = () => {
            alert('Error loading sample image. Please upload your own image instead.');
        };
        img.src = src;
    }

    // Draw image on canvas
    function drawImageOnCanvas(img) {
        // Calculate canvas size to maintain aspect ratio
        const containerWidth = canvas.parentElement.clientWidth;
        const maxHeight = window.innerHeight * 0.7;

        let canvasWidth = img.width;
        let canvasHeight = img.height;

        // Scale down if needed
        if (canvasWidth > containerWidth) {
            const scale = containerWidth / canvasWidth;
            canvasWidth = containerWidth;
            canvasHeight = img.height * scale;
        }

        // Further scale down if height exceeds maximum
        if (canvasHeight > maxHeight) {
            const scale = maxHeight / canvasHeight;
            canvasHeight = maxHeight;
            canvasWidth = canvasWidth * scale;
        }

        // Set canvas dimensions
        canvas.width = canvasWidth;
        canvas.height = canvasHeight;

        // Draw image
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvasWidth, canvasHeight);

        // Store canvas bounds for interaction
        canvasBounds = canvas.getBoundingClientRect();

        // Draw existing interaction points if any
        drawInteractionPoints();
    }

    // Get canvas point from mouse event
    function getCanvasPoint(e) {
        if (!canvasBounds) return { x: 0, y: 0 };

        const x = e.clientX - canvasBounds.left;
        const y = e.clientY - canvasBounds.top;

        return { x, y };
    }

    // Handle canvas click
    function handleCanvasClick(e) {
        if (!currentImage) return;

        const point = getCanvasPoint(e);

        if (promptType === 'point') {
            // For point prompt, immediately send request
            generateSegmentationForPoint(point);
        } else if (promptType === 'box') {
            // For box prompt, add point and maybe generate
            addBoxPoint(point);
        } else if (promptType === 'scribble') {
            // Scribble points are handled by mousedown/move events
            // This is just for single clicks
            addScribblePoint(point);
        }
    }

    // Add a point to the scribble
    function addScribblePoint(point) {
        scribblePoints.push(point);
        drawInteractionPoints();
        updateGenerateButtonState();
    }

    // Add a point to the box selection
    function addBoxPoint(point) {
        // If we already have 2 points, reset
        if (boxPoints.length >= 2) {
            boxPoints = [];
        }

        boxPoints.push(point);
        drawInteractionPoints();

        if (boxPoints.length === 2) {
            updateGenerateButtonState();
        }
    }

    // Draw the interaction points on the canvas
    function drawInteractionPoints() {
        if (!currentImage) return;

        // Redraw the image first
        ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);

        if (promptType === 'point') {
            // Nothing to draw for point prompt
        } else if (promptType === 'scribble') {
            // Draw scribble points and lines
            if (scribblePoints.length > 0) {
                ctx.beginPath();
                ctx.moveTo(scribblePoints[0].x, scribblePoints[0].y);

                for (let i = 1; i < scribblePoints.length; i++) {
                    ctx.lineTo(scribblePoints[i].x, scribblePoints[i].y);
                }

                ctx.strokeStyle = '#4CAF50';
                ctx.lineWidth = 3;
                ctx.stroke();

                // Draw points
                for (const point of scribblePoints) {
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, 3, 0, 2 * Math.PI);
                    ctx.fillStyle = '#4CAF50';
                    ctx.fill();
                }
            }
        } else if (promptType === 'box') {
            // Draw box points and rectangle
            for (const point of boxPoints) {
                ctx.beginPath();
                ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
                ctx.fillStyle = '#FF5722';
                ctx.fill();
            }

            if (boxPoints.length === 2) {
                const [p1, p2] = boxPoints;
                ctx.beginPath();
                ctx.rect(
                    Math.min(p1.x, p2.x),
                    Math.min(p1.y, p2.y),
                    Math.abs(p2.x - p1.x),
                    Math.abs(p2.y - p1.y)
                );
                ctx.strokeStyle = '#FF5722';
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        }
    }

    // Generate segmentation for a point
    function generateSegmentationForPoint(point) {
        if (!currentImage) return;

        // Convert canvas-relative point to image-relative
        const normalizedPoint = {
            x: point.x / canvas.width,
            y: point.y / canvas.height
        };

        // Prepare prompt data
        const promptData = {
            type: 'point',
            point: [normalizedPoint.x, normalizedPoint.y]
        };

        // Send request
        sendSegmentationRequest(promptData);
    }

    // Handle text submit
    function handleTextSubmit() {
        if (!currentImage) return;

        const text = textInput.value.trim();
        if (!text) {
            alert('Please enter text (e.g., "cat" or "dog")');
            return;
        }

        // Prepare prompt data
        const promptData = {
            type: 'text',
            text: text
        };

        // Send request
        sendSegmentationRequest(promptData);
    }

    // Generate segmentation based on current state
    function generateSegmentation() {
        if (!currentImage) return;

        let promptData;

        if (promptType === 'scribble') {
            if (scribblePoints.length < 2) {
                alert('Please draw at least 2 points');
                return;
            }

            // Normalize points
            const normalizedPoints = scribblePoints.map(point => [
                point.x / canvas.width,
                point.y / canvas.height
            ]);

            promptData = {
                type: 'scribble',
                points: normalizedPoints
            };
        } else if (promptType === 'box') {
            if (boxPoints.length !== 2) {
                alert('Please select two points to define a box');
                return;
            }

            // Normalize points
            const normalizedPoints = boxPoints.map(point => [
                point.x / canvas.width,
                point.y / canvas.height
            ]);

            promptData = {
                type: 'box',
                points: normalizedPoints
            };
        } else if (promptType === 'point') {
            // Use center point as default
            promptData = {
                type: 'point',
                point: [0.5, 0.5]
            };
        } else {
            // Text prompt handled separately
            return;
        }

        // Send request
        sendSegmentationRequest(promptData);
    }

    // Send segmentation request to server
    function sendSegmentationRequest(promptData) {
        if (!currentImage) return;

        // Show loading overlay
        loadingOverlay.style.display = 'flex';

        // Convert image to data URL if it's not already
        let imageDataUrl;
        if (currentImage.src.startsWith('data:')) {
            imageDataUrl = currentImage.src;
        } else {
            // Draw to canvas and get data URL
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = currentImage.width;
            tempCanvas.height = currentImage.height;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(currentImage, 0, 0);
            imageDataUrl = tempCanvas.toDataURL('image/jpeg');
        }

        // Send request to server
        fetch('/api/segment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: imageDataUrl,
                prompt: promptData
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }

            // Display the result
            displaySegmentationResult(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error generating segmentation: ' + error.message);
        })
        .finally(() => {
            // Hide loading overlay
            loadingOverlay.style.display = 'none';
        });
    }

    // Display segmentation result
    function displaySegmentationResult(data) {
        if (!data.segmentation_image) {
            console.error('No segmentation image in response');
            return;
        }

        // Create image from base64 data
        const img = new Image();
        img.onload = () => {
            // Replace current image with segmentation result
            currentImage = img;

            // Draw on canvas
            drawImageOnCanvas(img);

            // Reset interaction state
            resetInteractionState();
        };
        img.src = 'data:image/png;base64,' + data.segmentation_image;
    }

    // Reset canvas to original state
    function resetCanvas() {
        // Clear current image
        currentImage = null;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Reset interaction state
        resetInteractionState();

        // Update button state
        updateGenerateButtonState();
    }

    // Reset interaction state
    function resetInteractionState() {
        scribblePoints = [];
        boxPoints = [];
        isDrawing = false;

        // Update generate button
        updateGenerateButtonState();
    }

    // Initialize the app
    init();
});