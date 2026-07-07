import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, ViTImageProcessor, ViTModel
from PIL import Image

class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    @torch.no_grad()
    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        self.model.to(device)
        self.model.eval()
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        # Использование CLS токена
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu()

class VisionEncoder(nn.Module):
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        super().__init__()
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        
    @torch.no_grad()
    def forward(self, images: list[Image.Image], device: torch.device) -> torch.Tensor:
        self.model.to(device)
        self.model.eval()
        inputs = self.processor(images=images, return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        # Использование CLS токена
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu()
