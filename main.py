import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import os
import torch
import tempfile
import torchvision.transforms as transforms
import cv2
from torch import nn
from torchvision import models


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
    
VIDEO_DIR = "static/videos"
if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)
os.makedirs("chunks", exist_ok=True)

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

def load_model(path_to_model, num_classes=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(num_classes).to(device)
    checkpoint = torch.load(path_to_model, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def preprocess_video(video_path, sequence_length=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = train_transforms(frame)
        frames.append(frame)
    cap.release()
    if len(frames) < sequence_length:
        raise ValueError("Video too short")
    video_data = torch.stack(frames).unsqueeze(0)
    return video_data


def predict(model, video_data):
    device = next(model.parameters()).device
    video_data = video_data.to(device)
    with torch.no_grad():
        _, output = model(video_data)
        _, predicted = torch.max(output.data, 1)
        return predicted

model_path = "./model/checkpoint.pt"
model = load_model(model_path)

def predict_video(video_path):
    try:
        video_data = preprocess_video(video_path)
        prediction = predict(model, video_data)
        if prediction[0] == 1:
            return {"prediction": "REAL"}
        else:
            return {"prediction": "FAKE"}
    except Exception as e:
        return {"error": str(e)}


model = load_model(model_path)
@app.post("/save-video/")
async def save_video(file: UploadFile = File(...)):
    file_location = os.path.join(VIDEO_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"success": True, "file_path": file_location}

@app.get("/analyze-video")
async def analyze_video(file_path: str):
    prediction = predict_video(file_path)
    return {"result": prediction}


   

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), media_type="text/html")
