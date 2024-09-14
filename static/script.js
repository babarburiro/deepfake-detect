// Access buttons and status elements
const startButton = document.getElementById('start');
const stopButton = document.getElementById('stop');
const statusElement = document.getElementById('status');
const videoFileInput = document.getElementById('video-file');
const resultImage = document.getElementById("result-image");
const resultContainer = document.getElementById("result-container");
let mediaRecorder; // Global variable for MediaRecorder
let chunks = []; // Store video chunks

// Disable the stop button initially
stopButton.disabled = true;

startButton.addEventListener('click', async () => {
    resultContainer.style.display = 'None';
    try {
        // Disable the start button when clicked
        startButton.disabled = true;

        // Change status to "Recording"
        statusElement.textContent = "Recording...";

        // Start screen recording
        const stream = await navigator.mediaDevices.getDisplayMedia({
            video: { mediaSource: "screen" }
        });

        // If stream is successfully selected, enable stop button
        stopButton.disabled = false;

        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (e) => {
            chunks.push(e.data);
        };

        mediaRecorder.start();

        // When "Stop" button is pressed
        stopButton.addEventListener('click', () => {
            // Stop recording
            mediaRecorder.stop();

            // Disable stop button until processing is done
            stopButton.disabled = true;

            // Update status to "Processing"
            statusElement.textContent = "Processing...";

            // Combine the chunks into a single video blob
            mediaRecorder.onstop = async () => {
                const blob = new Blob(chunks, { type: 'video/webm' });
                const file = new File([blob], 'screen_recording.webm', { type: 'video/webm' });

                // Reset chunks for next recording
                chunks = [];

                // Create a form to send to the server
                const formData = new FormData();
                formData.append('file', file);
                stream.getTracks().forEach(track => track.stop());

                // Send to backend for processing
                try {
                    const response = await fetch('/save-video/', {
                        method: 'POST',
                        body: formData
                    });
                   
                    const result = await response.json();
        
                    if (result.success) {
                        // Call the API with the saved file path
                        fetch(`/analyze-video?file_path=${encodeURIComponent(result.file_path)}`, {
                            method: 'GET'
                        })
                        .then(response => response.json())
                        .then(data => {
                            // console.log(data.result)
                            statusElement.textContent = `Status: DeepFake Analysis Result: ${data.result}`;
                            if (data.img) {
                                resultImage.src = data.img;  // Assuming `img` is a URL or base64 string of the image
                                // Show result container
                                resultContainer.style.display = 'block';
                            }
                            statusElement.classList.remove('alert-warning');
                            statusElement.classList.add('alert-success');
                            startButton.disabled = false;
                        })
                        .catch(error => {
                            statusElement.textContent = 'Status: Error during DeepFake analysis.';
                            statusElement.classList.remove('alert-warning');
                            statusElement.classList.add('alert-danger');
                            console.error('Error:', error);
                            startButton.disabled = false;
                        });
                    } else {
                        statusElement.textContent = 'Status: Error saving video.';
                        statusElement.classList.remove('alert-warning');
                        statusElement.classList.add('alert-danger');
                        startButton.disabled = false;
                    }
                } catch (error) {
                    console.error('Error during processing:', error);
                    statusElement.textContent = "Error during processing";
                    startButton.disabled = false;
                }
            };
        });
    } catch (error) {
        // If the user cancels the screen recording, re-enable the start button
        startButton.disabled = false;
        statusElement.textContent = "Recording canceled!";
        console.error('Screen recording was not started:', error);
    }
});




// document.getElementById('start').onclick = async function() {
//     const statusElement = document.getElementById('status');
//     statusElement.textContent = 'Status: Recording...';
//     statusElement.classList.remove('alert-info', 'alert-success', 'alert-warning');
//     statusElement.classList.add('alert-info');
//     this.disabled = true;
//     const stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
//     const mediaRecorder = new MediaRecorder(stream);
//     let chunks = [];

//     mediaRecorder.ondataavailable = function(event) {
//         if (event.data.size > 0) {
//             chunks.push(event.data);
//         }
//     };

//     mediaRecorder.start();

//     document.getElementById('stop').onclick = function() {
//         mediaRecorder.stop();
//         statusElement.textContent = 'Status: Processing...';
//         statusElement.classList.remove('alert-info', 'alert-success', 'alert-warning');
//         statusElement.classList.add('alert-warning');

//         mediaRecorder.onstop = async function() {
//             const blob = new Blob(chunks, { type: 'video/webm' });
//             const file = new File([blob], `recording-${Date.now()}.webm`, { type: 'video/webm' });

//             const formData = new FormData();
//             formData.append('file', file);
//             stream.getTracks().forEach(track => track.stop());
//             // Save the video file locally
//             const response = await fetch('/save-video/', {
//                 method: 'POST',
//                 body: formData
//             });
           
//             const result = await response.json();

//             if (result.success) {
//                 // Call the API with the saved file path
//                 fetch(`/analyze-video?file_path=${encodeURIComponent(result.file_path)}`, {
//                     method: 'GET'
//                 })
//                 .then(response => response.json())
//                 .then(data => {
//                     // console.log(data.result)
//                     statusElement.textContent = `Status: DeepFake Analysis Result: ${data.result}`;
//                     if (data.img) {
//                         resultImage.src = data.img;  // Assuming `img` is a URL or base64 string of the image
//                         // Show result container
//                         resultContainer.style.display = 'block';
//                     }
//                     statusElement.classList.remove('alert-warning');
//                     statusElement.classList.add('alert-success');
//                 })
//                 .catch(error => {
//                     statusElement.textContent = 'Status: Error during DeepFake analysis.';
//                     statusElement.classList.remove('alert-warning');
//                     statusElement.classList.add('alert-danger');
//                     console.error('Error:', error);
//                 });
//             } else {
//                 statusElement.textContent = 'Status: Error saving video.';
//                 statusElement.classList.remove('alert-warning');
//                 statusElement.classList.add('alert-danger');
//             }
//         };
//     };
// };

document.getElementById('upload').onchange = async function(event) {
    const statusElement = document.getElementById('status');
    const file = event.target.files[0];
    const resultContainer = document.getElementById("result-container");
    const resultImage = document.getElementById("result-image");
    resultContainer.style.display = 'None';
    console.log('file')
    console.log(file)
    if (file) {
        statusElement.textContent = 'Status: Processing uploaded video...';
        statusElement.classList.remove('alert-info', 'alert-success', 'alert-warning', 'alert-danger');
        statusElement.classList.add('alert-warning');
        
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/save-video/', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            // Call the API with the saved file path
            fetch(`/analyze-video?file_path=${encodeURIComponent(result.file_path)}`, {
                method: 'GET'
            })
            .then(response => response.json())
            .then(data => {

                statusElement.textContent = `Status: DeepFake Analysis Result: ${data.result}`;
                if (data.img) {
                    resultImage.src = data.img;  // Assuming `img` is a URL or base64 string of the image
                    // Show result container
                    resultContainer.style.display = 'block';
                }
                statusElement.classList.remove('alert-warning');
                statusElement.classList.add('alert-success');
            })
            .catch(error => {
                statusElement.textContent = 'Status: Error during DeepFake analysis.';
                statusElement.classList.remove('alert-warning');
                statusElement.classList.add('alert-danger');
                console.error('Error:', error);
            });
        } else {
            statusElement.textContent = 'Status: Error saving uploaded video.';
            statusElement.classList.remove('alert-warning');
            statusElement.classList.add('alert-danger');
        }
    }
};