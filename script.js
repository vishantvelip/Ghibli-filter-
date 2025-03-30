// Load an image from a file input
function loadImage(file) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.src = URL.createObjectURL(file);
    });
}

// Create a mask from BodyPix segmentation
function createMask(segmentation, width, height) {
    const maskData = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < segmentation.data.length; i++) {
        maskData[i * 4 + 3] = segmentation.data[i] ? 255 : 0;
    }
    return new ImageData(maskData, width, height);
}

// Apply style transfer using the model
async function applyStyleTransfer(model, contentImg, styleImg) {
    const contentTensor = tf.browser.fromPixels(contentImg).toFloat().div(255).expandDims();
    const styleTensor = tf.browser.fromPixels(styleImg).toFloat().div(255).expandDims();
    const styledTensor = await model.predict([contentTensor, styleTensor]);
    const styledData = await tf.browser.toPixels(styledTensor.squeeze());
    return new ImageData(styledData, contentImg.width, contentImg.height);
}

// Composite the styled foreground onto the original background
function compositeImages(originalImageData, styledImageData, maskImageData) {
    const resultData = new Uint8ClampedArray(originalImageData.data);
    for (let i = 0; i < originalImageData.data.length; i += 4) {
        if (maskImageData.data[i + 3] > 0) {
            resultData[i] = styledImageData.data[i];
            resultData[i + 1] = styledImageData.data[i + 1];
            resultData[i + 2] = styledImageData.data[i + 2];
            resultData[i + 3] = styledImageData.data[i + 3];
        }
    }
    return new ImageData(resultData, originalImageData.width, originalImageData.height);
}

// Main processing function
async function processImage() {
    try {
        const processButton = document.getElementById('process-button');
        processButton.disabled = true;
        processButton.textContent = 'Processing...';

        const photoFile = document.getElementById('photo-input').files[0];
        const styleFile = document.getElementById('style-input').files[0];
        if (!photoFile || !styleFile) {
            alert('Please upload both a photo and a style image.');
            return;
        }

        const photoImg = await loadImage(photoFile);
        const styleImg = await loadImage(styleFile);

        const photoCanvas = document.createElement('canvas');
        photoCanvas.width = photoImg.width;
        photoCanvas.height = photoImg.height;
        const photoCtx = photoCanvas.getContext('2d');
        photoCtx.drawImage(photoImg, 0, 0);
        const photoImageData = photoCtx.getImageData(0, 0, photoImg.width, photoImg.height);

        const net = await bodyPix.load();
        const segmentation = await net.segmentPerson(photoImg);
        const maskImageData = createMask(segmentation, photoImg.width, photoImg.height);

        const model = await tf.loadGraphModel(
            'https://tfhub.dev/google/tfjs-model/magenta/arbitrary-image-stylization-v1-256/1/default/1',
            { fromTFHub: true }
        );

        const styledImageData = await applyStyleTransfer(model, photoImg, styleImg);
        const resultImageData = compositeImages(photoImageData, styledImageData, maskImageData);

        const canvas = document.getElementById('result-canvas');
        canvas.width = photoImg.width;
        canvas.height = photoImg.height;
        const ctx = canvas.getContext('2d');
        ctx.putImageData(resultImageData, 0, 0);

    } catch (error) {
        console.error('Error processing image:', error);
        alert('An error occurred while processing the image.');
    } finally {
        const processButton = document.getElementById('process-button');
        processButton.disabled = false;
        processButton.textContent = 'Process Image';
    }
}

// Preview functions
function previewPhoto() {
    const file = document.getElementById('photo-input').files[0];
    if (file) console.log('Photo selected:', file.name);
}

function previewStyle() {
    const file = document.getElementById('style-input').files[0];
    if (file) console.log('Style image selected:', file.name);
}

document.getElementById('process-button').addEventListener('click', processImage);

