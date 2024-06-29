import torch
from torchvision.models import resnet50
from transformers import CLIPModel
from tokenizers import ByteLevelBPETokenizer

from model.loss import ContrastiveLoss
from model.fusion_model import CrossAttentionFusionEncoder
from dataload.dataloader import create_data_loaders

# Fine-tuning loop
def fine_tune_encoder(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device, model_save_path):
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for cnn_images, clip_images, tokens in train_loader:
            cnn_images = cnn_images.to(device)
            clip_images = clip_images.to(device)
            tokens = tokens.to(device)

            # Forward pass
            image_features = model(cnn_images, clip_images)

            # Compute loss
            loss = criterion(image_features, tokens)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss}")

        # Validation phase
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for cnn_images, clip_images, tokens in valid_loader:
                cnn_images = cnn_images.to(device)
                clip_images = clip_images.to(device)
                tokens = tokens.to(device)

                # Forward pass
                image_features = model(cnn_images, clip_images)

                # Compute loss
                loss = criterion(image_features, tokens)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {valid_loss}")

        # Save the best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Epoch [{epoch+1}/{num_epochs}], Saved Best Model with Validation Loss: {valid_loss}")

if __name__ == "__main__":
    # Check if CUDA is available and set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load the custom tokenizer
    tokenizer = ByteLevelBPETokenizer(
        '/home/dasith/Documents/Personal/Academics/chestXtray/Pytorch_impl/preprocess/mimic-vocab.json',
        '/home/dasith/Documents/Personal/Academics/chestXtray/Pytorch_impl/preprocess/mimic-merges.txt',
    )

    # Paths
    csv_file = '/home/dasith/Documents/Personal/Academics/chestXtray/Datasets/indiana/cleaned_df.csv'

    # Create data loaders
    train_loader, valid_loader = create_data_loaders(csv_file, tokenizer, batch_size=16)

    # Initialize models
    cnn_backbone = resnet50(pretrained=True)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    fusion_model = CrossAttentionFusionEncoder(cnn_backbone, clip_model, embed_dim=256).to(device)

    # Initialize contrastive loss
    contrastive_loss = ContrastiveLoss()

    # Define an optimizer
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

    # Define model save path
    model_save_path = '/home/dasith/Documents/Personal/Academics/chestXtray/chestxpert_fusion/checkpoints/best_fusion_model.pth'

    # Fine-tune the encoder
    fine_tune_encoder(fusion_model, train_loader, valid_loader, contrastive_loss, optimizer, num_epochs=10, device=device, model_save_path=model_save_path)
