import torch
from torchvision.models import resnet50
from transformers import CLIPModel
from tokenizers import ByteLevelBPETokenizer
from torch import nn
import os

from model.loss import ContrastiveLoss
from model.fused_encoder import CrossAttentionFusionEncoder
from model.decoder import Decoder
from model.end2end import Transformer
from model.utils import create_target_masks

from dataload.encoder.dataloader import create_data_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

cnn_backbone = resnet50(pretrained=True)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
encoder = CrossAttentionFusionEncoder(cnn_backbone, clip_model, embed_dim=256)

# Load pre-trained weights
encoder.load_state_dict(torch.load('/home/dasith/Documents/Personal/Academics/chestXpert/chestxpert_fusion/checkpoints/best_fusion_model_3.pth'))

# Ensure the encoder is frozen
for param in encoder.parameters():
    param.requires_grad = False

num_layers = 6
d_model = 256
dff = 2048
num_heads = 8
dropout_rate = 0.1

# Load the custom tokenizer
tokenizer = ByteLevelBPETokenizer(
    '/home/dasith/Documents/Personal/Academics/chestXpert/chestxpert_fusion/preprocess/mimic-vocab.json',
    '/home/dasith/Documents/Personal/Academics/chestXpert/chestxpert_fusion/preprocess/mimic-merges.txt',
)

# Paths
csv_file = '/home/dasith/Documents/Personal/Academics/chestXpert/Datasets/indiana/cleaned_df.csv'

# Create data loaders
train_loader, valid_loader = create_data_loaders(csv_file, tokenizer, batch_size=16)

target_vocab_size = tokenizer.get_vocab_size()

decoder = Decoder(num_layers, d_model, num_heads, dff,
                  target_vocab_size, target_vocab_size, rate=dropout_rate, device=device)

# Define the encoder-decoder model
model = Transformer(encoder, decoder, device, d_model, target_vocab_size).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding index
optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
best_val_loss = float('inf')
checkpoint_dir = '/home/dasith/Documents/Personal/Academics/chestXpert/chestxpert_fusion/checkpoints/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for cnn_images, clip_images, captions in train_loader:
        cnn_images = cnn_images.to(device)
        clip_images = clip_images.to(device)
        captions = captions.to(device)

        caption_inp = captions[:, :-1]
        caption_real = captions[:, 1:].reshape(-1).long()

        # Create padding masks for input/target
        with torch.no_grad():
            combined_mask = create_target_masks(caption_inp)

        # Forward pass
        dec_output, attn_weights = model(cnn_images, clip_images, caption_inp, True, combined_mask, None)  # Exclude the last token for input
        loss = criterion(dec_output.view(-1, target_vocab_size), caption_real)  # Shift by one for target

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for cnn_images, clip_images, captions in valid_loader:
            cnn_images = cnn_images.to(device)
            clip_images = clip_images.to(device)
            captions = captions.to(device)

            caption_inp = captions[:, :-1]
            caption_real = captions[:, 1:].reshape(-1).long()

            # Create padding masks for input/target
            combined_mask = create_target_masks(caption_inp)

            # Forward pass
            dec_output, attn_weights = model(cnn_images, clip_images, caption_inp, False, combined_mask, None)  # Exclude the last token for input
            loss = criterion(dec_output.view(-1, target_vocab_size), caption_real)  # Shift by one for target

            val_loss += loss.item()

    avg_val_loss = val_loss / len(valid_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save the model if validation loss decreases
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model_save_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f'Saved best model to {model_save_path}')

print("Training and validation completed.")
