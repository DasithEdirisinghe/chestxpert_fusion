import torch
from torchvision.models import resnet50
from transformers import CLIPModel
from tokenizers import ByteLevelBPETokenizer
import os

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

# Create validation data loader
_, valid_loader = create_data_loaders(csv_file, tokenizer, batch_size=16)

target_vocab_size = tokenizer.get_vocab_size()

decoder = Decoder(num_layers, d_model, num_heads, dff,
                  target_vocab_size, target_vocab_size, rate=dropout_rate, device=device)

# Define the encoder-decoder model
model = Transformer(encoder, decoder, device, d_model, target_vocab_size).to(device)

# Load the saved model weights
model_path = '/home/dasith/Documents/Personal/Academics/chestXpert/chestxpert_fusion/checkpoints/best_model_epoch_6.pth'
model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
model.eval()

# Inference loop
generated_texts = []
actual_texts = []
max_length = 128  # Set an appropriate max length for your task

with torch.no_grad():
    for cnn_images, clip_images, captions in valid_loader:
        cnn_images = cnn_images.to(device)
        clip_images = clip_images.to(device)
        captions = captions.to(device)

        # Decode actual captions
        for caption in captions:
            actual_text = tokenizer.decode([int(token) for token in caption.tolist()], skip_special_tokens=True)
            actual_texts.append(actual_text)

        # Initial input token (start token)
        input_ids = torch.tensor([[tokenizer.token_to_id('<s>')]]).to(device)

        # Generate text
        for _ in range(max_length):
            combined_mask = create_target_masks(input_ids)

            dec_output, attn_weights = model(cnn_images, clip_images, input_ids, False, combined_mask, None)
            predictions = dec_output[:, -1:, :]

            predicted_id = torch.argmax(predictions, dim=-1)

            # Concatenate predicted_id to input_ids for next iteration
            input_ids = torch.cat([input_ids, predicted_id], dim=-1)

            # Break if end token is generated
            if predicted_id == tokenizer.token_to_id('</s>'):
                break

        # Decode generated text
        generated_text = tokenizer.decode(input_ids.squeeze().tolist(), skip_special_tokens=True)
        generated_texts.append(generated_text)

        # Print or save the actual and generated texts
        for i, (actual, generated) in enumerate(zip(actual_texts, generated_texts)):
            print(f"Actual Text {i + 1}: {actual}")
            print(f"Generated Text {i + 1}: {generated}\n")

# Print or save the actual and generated texts
for i, (actual, generated) in enumerate(zip(actual_texts, generated_texts)):
    print(f"Actual Text {i + 1}: {actual}")
    print(f"Generated Text {i + 1}: {generated}\n")

print("Inference completed.")