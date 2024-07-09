import torch
from model.decoder import Decoder
from model.fused_encoder import CrossAttentionFusionEncoder

class Transformer(torch.nn.Module):
    def __init__(self, encoder, decoder, device, d_model, target_vocab_size):
        super(Transformer, self).__init__()

        self.encoder = encoder

        self.decoder = decoder

        self.final_layer = torch.nn.Linear(d_model, target_vocab_size).to(device=device)

    def forward(self, cnn_inp, clip_inp, tar, training, look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(cnn_inp, clip_inp)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights