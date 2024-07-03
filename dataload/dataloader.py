from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from transformers import AutoProcessor

import pandas as pd
import torch
import numpy as np

from dataload.datasets import ImageCaptionDataset

def create_data_loaders(csv_file, tokenizer, batch_size, valid_size=0.2, shuffle=True, random_seed=42):
    # Read the CSV file
    data_frame = pd.read_csv(csv_file)

    # Ensure reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Split the dataset
    train_df, valid_df = train_test_split(data_frame, test_size=valid_size, random_state=random_seed)

    # Initialize the CLIP processor
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Create datasets
    train_dataset = ImageCaptionDataset(data_frame=train_df, tokenizer=tokenizer, clip_processor=clip_processor)
    valid_dataset = ImageCaptionDataset(data_frame=valid_df, tokenizer=tokenizer, clip_processor=clip_processor)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader

