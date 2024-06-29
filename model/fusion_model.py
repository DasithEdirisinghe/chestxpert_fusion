import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusionEncoder(nn.Module):
    def __init__(self, cnn_backbone, clip_model, embed_dim):
        super(CrossAttentionFusionEncoder, self).__init__()
        self.cnn_backbone = cnn_backbone
        self.clip_model = clip_model

        # Remove the classification head of the CNN backbone
        self.cnn_backbone = nn.Sequential(*list(cnn_backbone.children())[:-2])

        # Freeze the CNN backbone
        for param in self.cnn_backbone.parameters():
            param.requires_grad = False

        # Freeze the CLIP image encoder
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Linear layers to project CNN and CLIP features to a common dimension
        self.cnn_proj = nn.Linear(2048, embed_dim)  # Assuming ResNet-50 output is 2048
        self.clip_proj = nn.Linear(512, embed_dim)   # Assuming CLIP's output is 512

        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8)

    def forward(self, cnn_images, clip_images):
        # Extract features from CNN backbone
        cnn_features = self.cnn_backbone(cnn_images)
        cnn_features = F.adaptive_avg_pool2d(cnn_features, (1, 1)).squeeze()

        if cnn_features.dim() == 1:
            cnn_features = cnn_features.unsqueeze(0)

        cnn_features = self.cnn_proj(cnn_features)

        # Extract features from CLIP image encoder
        clip_image_features = self.clip_model.get_image_features(clip_images)
        clip_image_features = self.clip_proj(clip_image_features)

       # Reshape for cross-attention (batch_size, seq_len, embed_dim)
        cnn_features = cnn_features.unsqueeze(1)   # (batch_size, 1, embed_dim)
        clip_image_features = clip_image_features.unsqueeze(1)   # (batch_size, 1, embed_dim)
        # Apply cross-attention
        cross_attended, _ = self.cross_attention(cnn_features, clip_image_features, clip_image_features)  # (batch_size, 1, embed_dim)

        # Use cross-attended features
        return cross_attended.squeeze(1)  # (batch_size, embed_dim)
