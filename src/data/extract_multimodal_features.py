from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.models.multimodal_encoders import TextEncoder, VisionEncoder


def extract_features(data_dir: Path, output_dir: Path, batch_size: int = 32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используем устройство: {device}")
    
    csv_path = data_dir / "MM-ML-1M-main" / "movies_details_clean.csv"
    posters_dir = data_dir / "MM-ML-1M-main" / "posters"
    
    print("Загрузка данных...")
    df = pd.read_csv(csv_path)
    
    # Инициализация энкодеров
    print("Инициализация энкодеров...")
    text_encoder = TextEncoder().to(device)
    vision_encoder = VisionEncoder().to(device)
    
    text_embeddings = {}
    vision_embeddings = {}
    
    # Обработка батчами
    for i in tqdm(range(0, len(df), batch_size), desc="Извлечение мультимодальных признаков"):
        batch = df.iloc[i:i+batch_size]
        
        # Текст
        texts = batch['Overview'].fillna("").tolist()
        batch_text_embs = text_encoder(texts, device)
        
        # Изображения
        images = []
        valid_indices = []
        for idx, row in batch.iterrows():
            # Простой способ найти постер по Id
            poster_files = list(posters_dir.glob(f"{row['Id']}_*.jpg"))

            img_loaded = False
            if poster_files:
                try:
                    img = Image.open(poster_files[0]).convert("RGB")
                    images.append(img)
                    valid_indices.append(idx)
                    img_loaded = True
                except (OSError, ValueError):
                    img_loaded = False
                    
            if not img_loaded:
                # Dummy image
                images.append(Image.new("RGB", (224, 224), (255, 255, 255)))
                
        batch_vision_embs = vision_encoder(images, device)
        
        for k, (_, row) in enumerate(batch.iterrows()):
            item_id = row['Id']
            text_embeddings[item_id] = batch_text_embs[k].numpy()
            vision_embeddings[item_id] = batch_vision_embs[k].numpy()
            
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем как словари
    print("Сохранение признаков...")
    torch.save(text_embeddings, output_dir / "text_embeddings.pt")
    torch.save(vision_embeddings, output_dir / "vision_embeddings.pt")
    print("Готово!")

if __name__ == '__main__':
    extract_features(Path("data/MM-ML-1M"), Path("artifacts/models"))
