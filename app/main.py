from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as T
import io
import os

from model import IntrinsicNet
from relight import relight

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/web", StaticFiles(directory="web"), name="web")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.get("/", response_class=HTMLResponse)
def home():
    return open("web/index.html", encoding="utf-8").read()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = IntrinsicNet().to(device)
model.eval()

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

@app.post("/api/relight")
async def relight_api(
    image: UploadFile = File(...),
    direction: str = Form(...),
    brightness: int = Form(...)
):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        normals, diffuse, specular = model(x)

        light_dirs = {
            "front": [0, 0, 1],
            "side": [1, 0, 1],
            "back": [-1, 0, 1]
        }

        out = relight(
            normals,
            diffuse,
            specular,
            light_dirs.get(direction, [0, 0, 1]),
            brightness / 100.0
        )

    os.makedirs("outputs", exist_ok=True)
    out_img = T.ToPILImage()(out.squeeze().cpu())
    out_img.save("outputs/result.png")

    return {"url": "/outputs/result.png"}
