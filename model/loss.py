import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute cosine similarity
        logits_per_image = torch.matmul(image_features, text_features.T) / self.temperature
        logits_per_text = torch.matmul(text_features, image_features.T) / self.temperature

        # Create labels (same for both directions)
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size).long().to(image_features.device)

        # Calculate cross-entropy loss
        loss_image_to_text = F.cross_entropy(logits_per_image, labels)
        loss_text_to_image = F.cross_entropy(logits_per_text, labels)
        loss = (loss_image_to_text + loss_text_to_image) / 2

        return loss
