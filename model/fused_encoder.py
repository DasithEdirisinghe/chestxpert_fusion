import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusionEncoder(nn.Module):
    def __init__(self, cnn_backbone, clip_model, embed_dim, dropout_prob=0.1):
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
        self.cnn_proj = nn.Sequential(
            nn.Linear(2048, embed_dim),  # Assuming ResNet-50 output is 2048
            nn.BatchNorm1d(embed_dim),   # Batch normalization
            nn.ReLU(),                   # ReLU activation
            nn.Dropout(dropout_prob)     # Dropout for regularization
        )
        self.clip_proj = nn.Sequential(
            nn.Linear(512, embed_dim),   # Assuming CLIP's output is 512
            nn.BatchNorm1d(embed_dim),   # Batch normalization
            nn.ReLU(),                   # ReLU activation
            nn.Dropout(dropout_prob)     # Dropout for regularization
        )

        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8)

    def forward(self, cnn_images, clip_images):
        # Extract features from CNN backbone
        cnn_features = self.cnn_backbone(cnn_images)
        # Assuming the output shape is (batch_size, 2048, h, w) after CNN backbone
        # Flatten the spatial dimensions and project to embedding dimension
        batch_size, num_channels, h, w = cnn_features.shape
        cnn_features = cnn_features.flatten(2).permute(0, 2, 1)  # (batch_size, h*w, 2048)
        cnn_features = cnn_features.reshape(-1, num_channels)
        cnn_features = self.cnn_proj(cnn_features)  # (batch_size*h*w, embed_dim)
        cnn_features = cnn_features.reshape(batch_size, -1, cnn_features.size(-1))  # (batch_size, h*w, embed_dim)

        # Extract features from CLIP image encoder
        clip_image_features = self.clip_model.get_image_features(clip_images)
        # Assuming the output shape is (batch_size, 512)
        # Repeat features to match the sequence length
        clip_image_features = clip_image_features.unsqueeze(1).repeat(1, h * w, 1)  # (batch_size, h*w, 512)
        clip_image_features = clip_image_features.reshape(-1, clip_image_features.size(-1))
        clip_image_features = self.clip_proj(clip_image_features)  # (batch_size*h*w, embed_dim)
        clip_image_features = clip_image_features.reshape(batch_size, -1, clip_image_features.size(-1))  # (batch_size, h*w, embed_dim)

        # Apply cross-attention
        cross_attended, _ = self.cross_attention(cnn_features, clip_image_features, clip_image_features)  # (batch_size, h*w, embed_dim)

        # Aggregate cross-attended features to match the text feature dimension
        cross_attended = cross_attended.mean(dim=1)  # (batch_size, embed_dim)

        return cross_attended  # (batch_size, embed_dim)
