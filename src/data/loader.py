from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class MovieLensDataset(Dataset):
    """PyTorch Dataset wrapper for MovieLens preprocessed split data."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame containing preprocessed MovieLens data.
        """
        self.df = df.reset_index(drop=True)
        
        # Convert columns to PyTorch tensors where appropriate
        # Basic identifiers
        self.user_ids = torch.tensor(self.df["user_id"].values, dtype=torch.long)
        self.movie_ids = torch.tensor(self.df["movie_id"].values, dtype=torch.long)
        self.ratings = torch.tensor(self.df["rating"].values, dtype=torch.float32)
        self.timestamps = torch.tensor(self.df["timestamp"].values, dtype=torch.long)
        
        # We can extract user and movie metadata as lists/tensors
        self.gender = self.df["gender"].tolist()
        self.age = torch.tensor(self.df["age"].values, dtype=torch.long)
        self.occupation = torch.tensor(self.df["occupation"].values, dtype=torch.long)
        self.zip_code = self.df["zip_code"].tolist()
        self.title = self.df["title"].tolist()
        self.genres = self.df["genres"].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "user_id": self.user_ids[idx],
            "movie_id": self.movie_ids[idx],
            "rating": self.ratings[idx],
            "timestamp": self.timestamps[idx],
            "gender": self.gender[idx],
            "age": self.age[idx],
            "occupation": self.occupation[idx],
            "zip_code": self.zip_code[idx],
            "title": self.title[idx],
            "genres": self.genres[idx]
        }


class MovieLensDataLoader:
    """Utility class to load MovieLens processed splits and create PyTorch DataLoaders."""
    
    def __init__(self, data_dir: str = "data/processed"):
        """
        Args:
            data_dir: Path to the directory where train.csv, val.csv, and test.csv are saved.
        """
        self.data_dir = Path(data_dir)
        self.train_path = self.data_dir / "train.csv"
        self.val_path = self.data_dir / "val.csv"
        self.test_path = self.data_dir / "test.csv"

    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the train, validation, and test splits as pandas DataFrames.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if not (self.train_path.exists() and self.val_path.exists() and self.test_path.exists()):
            raise FileNotFoundError(
                f"Processed split files not found in '{self.data_dir}'. "
                f"Please run the preprocessing script first."
            )
            
        train = pd.read_csv(self.train_path)
        val = pd.read_csv(self.val_path)
        test = pd.read_csv(self.test_path)
        return train, val, test

    def get_pytorch_datasets(self) -> Tuple[MovieLensDataset, MovieLensDataset, MovieLensDataset]:
        """Load the splits and return PyTorch Dataset instances.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        train_df, val_df, test_df = self.load_splits()
        return (
            MovieLensDataset(train_df),
            MovieLensDataset(val_df),
            MovieLensDataset(test_df)
        )

    def get_pytorch_loaders(
        self,
        batch_size: int = 256,
        shuffle_train: bool = True,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load the splits and return PyTorch DataLoader instances.
        
        Args:
            batch_size: Size of batches for data loaders.
            shuffle_train: Whether to shuffle the training set.
            num_workers: Number of subprocesses to use for data loading.
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_dataset, val_dataset, test_dataset = self.get_pytorch_datasets()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_loader, val_loader, test_loader
