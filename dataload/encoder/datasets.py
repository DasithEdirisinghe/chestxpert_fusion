from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import torch.nn.functional as F
import torch
import numpy as np
import cv2

class ImageCaptionDataset(Dataset):
    def __init__(self, data_frame, tokenizer, clip_processor=None, max_length=256, mode='train'):

        assert mode in ['train', 'validate', 'test']

        self.data_frame = data_frame
        self.tokenizer = tokenizer
        self.clip_processor = clip_processor
        self.max_length = max_length
        self.mode = mode
    
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx, 1]  # Assumes the second column contains image paths
        caption = self.data_frame.iloc[idx, 2]  # Assumes the third column contains captions

         # Read image using PIL
        image = Image.open(img_path).convert('L')
        # Replicate grayscale image to create a 3-channel image
        image = Image.merge('RGB', (image, image, image))

        # Apply augmentations in training mode
        if self.mode == 'train':
            image = self.augment_image(image)

        # Preprocess image for CLIP
        clip_image = self.clip_processor(images=image, do_convert_rgb = False, do_normalize = True, return_tensors="pt")["pixel_values"].squeeze(0)

        # Convert image to numpy array for padding and resizing
        image_np = np.array(image)
        padding = self.get_padding(image_np)
        cnn_image = np.pad(image_np, padding, mode='constant', constant_values=0)
        cnn_image = cv2.resize(cnn_image, (224, 224), interpolation=cv2.INTER_NEAREST)
        cnn_image = np.divide(cnn_image, 255.).astype('float32')

        # Convert back to tensor and permute to (C, H, W)
        cnn_image = torch.tensor(cnn_image).permute(2, 0, 1)


        # Tokenize and pad the caption
        tokens = [self.tokenizer.token_to_id('<s>')] + self.tokenizer.encode(caption).ids + [self.tokenizer.token_to_id('</s>')]
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length-1] + [self.tokenizer.token_to_id('</s>')]
        token_tensor = torch.tensor(tokens, dtype=torch.float)
        token_tensor = F.pad(token_tensor, (0, self.max_length - len(token_tensor)), value=self.tokenizer.token_to_id('<pad>'))

        return cnn_image, clip_image, token_tensor

    def get_padding(self, image):
        height, width, channels = image.shape
        max_length = max(height, width)
        h_padding = (max_length - height) // 2
        v_padding = (max_length - width) // 2
        padding = ((h_padding, h_padding), (v_padding, v_padding), (0, 0))  # Pad height, width, and no padding for channels
        return padding
    
    def augment_image(self, image):
        # Define the augmentation transform
        augmentation_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.125, saturation=0.5, contrast=0.5),
            transforms.RandomHorizontalFlip()
        ])
        # Apply the augmentation
        image = augmentation_transform(image)
        
        return image