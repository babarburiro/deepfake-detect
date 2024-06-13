from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import os

app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Path for the combined video file
combined_video_path = "chunks/combined_video.webm"

# Ensure the chunks directory exists
os.makedirs("chunks", exist_ok=True)

@app.post("/upload-chunk/")
async def upload_chunk(files: List[UploadFile] = File(...)):
    with open(combined_video_path, "ab") as combined_video:
        for file in files:
            contents = await file.read()
            combined_video.write(contents)
    return JSONResponse(content={"message": "Chunk uploaded and combined successfully"})

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), media_type="text/html")
