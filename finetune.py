import torch
from torchvision.models import resnet50
from transformers import CLIPModel
from tokenizers import ByteLevelBPETokenizer

from model.loss import ContrastiveLoss
from model.fusion_model import CrossAttentionFusionEncoder
from dataload.dataloader import create_data_loaders

# Load the custom tokenizer
tokenizer = ByteLevelBPETokenizer(
        '/home/dasith/Documents/Personal/Academics/chestXtray/Pytorch_impl/preprocess/mimic-vocab.json',
        '/home/dasith/Documents/Personal/Academics/chestXtray/Pytorch_impl/preprocess/mimic-merges.txt',
    )

# Paths
csv_file = '/home/dasith/Documents/Personal/Academics/chestXtray/Datasets/indiana/cleaned_df.csv'

# Create data loaders
train_loader, valid_loader = create_data_loaders(csv_file, tokenizer, batch_size=16)

# Example usage
cnn_backbone = resnet50(pretrained=True)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
fusion_model = CrossAttentionFusionEncoder(cnn_backbone, clip_model, embed_dim=256)

# Initialize contrastive loss
contrastive_loss = ContrastiveLoss()

# Define an optimizer
optimizer = torch.optim.Adam(fusion_model.parameters(), lr=1e-4)


# Fine-tuning loop
def fine_tune_encoder(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for cnn_images, clip_images, tokens in dataloader:  # Assuming dataloader provides images and captions
            
            # Forward pass
            image_features = model(cnn_images, clip_images)

            # Compute loss
            loss = criterion(image_features, tokens)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")


fine_tune_encoder(fusion_model, train_loader, contrastive_loss, optimizer, num_epochs = 10)