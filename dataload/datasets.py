import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

class ImageCaptionDataset(Dataset):
    def __init__(self, data_frame, tokenizer, transform=None, clip_processor=None, max_length=256):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        self.transform = transform
        self.clip_processor = clip_processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx, 1]  # Assumes the second column contains image paths
        image = Image.open(img_path).convert('RGB')
        caption = self.data_frame.iloc[idx, 2]  # Assumes the third column contains captions

        # Preprocess image for CLIP
        clip_image = self.clip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        
        if self.transform:
            image = self.transform(image)

        # Tokenize and pad the caption
        tokens = [self.tokenizer.token_to_id('<s>')] + self.tokenizer.encode(caption).ids + [self.tokenizer.token_to_id('</s>')]
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length-1] + [self.tokenizer.token_to_id('</s>')]
        token_tensor = torch.tensor(tokens, dtype=torch.float)
        token_tensor = F.pad(token_tensor, (0, self.max_length - len(token_tensor)), value=self.tokenizer.token_to_id('<pad>'))

        return image, clip_image, token_tensor