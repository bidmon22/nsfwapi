from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

torch.set_num_threads(4)

app = FastAPI(title="NSFW Checker for iimg.live (Low RAM - Marqo)")

# CORS for your site
origins = [
    "https://iimg.live",
    "http://iimg.live",
    "https://*.iimg.live",
    "http://*.iimg.live",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load lightweight Marqo ViT-tiny model (~22MB weights, <150MB RAM)
model = timm.create_model("hf_hub:Marqo/nsfw-image-detection-384", pretrained=True)
model.eval()

# Get correct preprocessing transforms
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

# Warm-up the model
with torch.no_grad():
    dummy_img = Image.new("RGB", (384, 384))
    _ = model(transform(dummy_img).unsqueeze(0))

@app.get("/")
def root():
    return {"message": "Low-RAM NSFW checker API (Marqo model) is running!"}

@app.post("/check")
async def check_nsfw(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")

    # Apply model-specific preprocessing (includes resize to 384x384 + normalization)
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor).softmax(dim=-1)[0].cpu().numpy()

    # Class order: [SFW, NSFW]
    nsfw_score = float(output[1])
    sfw_score = float(output[0])
    label = "NSFW" if nsfw_score > sfw_score else "SFW"
    safe = label == "SFW"

    return {
        "label": label,
        "score": round(nsfw_score, 4),
        "safe": safe
    }
