import os
import zipfile
import requests
from pathlib import Path

def download_movielens(data_dir: Path) -> Path:
    """
    Скачивает датасет MovieLens-1M, если его еще нет.
    Возвращает путь к распакованной папке ml-1m.
    """
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = raw_dir / "ml-1m.zip"
    extract_dir = raw_dir / "ml-1m"
    
    if extract_dir.exists() and (extract_dir / "ratings.dat").exists():
        print("MovieLens-1M уже скачан и распакован.")
        return extract_dir
        
    print(f"Скачивание {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    print("Распаковка архива...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(raw_dir)
        
    print("Готово!")
    return extract_dir

if __name__ == '__main__':
    data_dir = Path("data")
    download_movielens(data_dir)
