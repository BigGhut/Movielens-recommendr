import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Define and save dummy PyTorch model state dicts
    dummy_user_weights = {
        "embedding.weight": torch.randn(10, 4),
        "fc.weight": torch.randn(4, 4),
        "fc.bias": torch.randn(4)
    }
    
    dummy_item_weights = {
        "embedding.weight": torch.randn(100, 4),
        "fc.weight": torch.randn(4, 4),
        "fc.bias": torch.randn(4)
    }
    
    torch.save(dummy_user_weights, model_dir / "user_tower.pt")
    torch.save(dummy_item_weights, model_dir / "item_tower.pt")
    
    # Save a mock FAISS index file (binary dummy content representing index)
    # The file contains a recognizable mock FAISS header
    mock_faiss_data = b"MOCK_FAISS_INDEX_DATA_WITH_VECTORS_SIZE_4"
    with open(model_dir / "item_index.faiss", "wb") as f:
        f.write(mock_faiss_data)
        
    print(f"Mock retrieval training complete. Saved models/user_tower.pt, models/item_tower.pt, and models/item_index.faiss in {model_dir}")

if __name__ == "__main__":
    main()
