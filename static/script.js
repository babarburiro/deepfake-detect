document.getElementById('start').onclick = async function() {
    const stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
    const mediaRecorder = new MediaRecorder(stream);
    let chunks = [];
    mediaRecorder.ondataavailable = function(event) {
        if (event.data.size > 0) {
            chunks.push(event.data);
            sendChunk(event.data);
        }
    };
    mediaRecorder.start(1000); // Collect data in 1 second chunks

    document.getElementById('stop').onclick = function() {
        mediaRecorder.stop();
    };

    function sendChunk(chunk) {
        const formData = new FormData();
        const filename = `chunk-${Date.now()}.webm`;
        formData.append('files', chunk, filename);
        
        fetch('/upload-chunk/', {
            method: 'POST',
            body: formData
        }).then(response => response.json())
          .then(result => console.log('Success:', result))
          .catch(error => console.error('Error:', error));
    }
};
