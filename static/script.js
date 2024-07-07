document.getElementById('start').onclick = async function() {
    const statusElement = document.getElementById('status');
    statusElement.textContent = 'Status: Recording...';
    statusElement.classList.remove('alert-info', 'alert-success', 'alert-warning');
    statusElement.classList.add('alert-info');

    const stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
    const mediaRecorder = new MediaRecorder(stream);
    let chunks = [];

    mediaRecorder.ondataavailable = function(event) {
        if (event.data.size > 0) {
            chunks.push(event.data);
        }
    };

    mediaRecorder.start();

    document.getElementById('stop').onclick = function() {
        mediaRecorder.stop();
        statusElement.textContent = 'Status: Processing...';
        statusElement.classList.remove('alert-info', 'alert-success', 'alert-warning');
        statusElement.classList.add('alert-warning');

        mediaRecorder.onstop = async function() {
            const blob = new Blob(chunks, { type: 'video/webm' });
            const file = new File([blob], `recording-${Date.now()}.webm`, { type: 'video/webm' });

            const formData = new FormData();
            formData.append('file', file);
            stream.getTracks().forEach(track => track.stop());
            // Save the video file locally
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
                    statusElement.textContent = `Status: DeepFake Analysis Result: ${data.result.prediction}`;
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
                statusElement.textContent = 'Status: Error saving video.';
                statusElement.classList.remove('alert-warning');
                statusElement.classList.add('alert-danger');
            }
        };
    };
};
