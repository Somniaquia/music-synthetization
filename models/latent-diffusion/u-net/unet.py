import torch
import torch.nn as nn
import pytorch_lightning as pl

from ..blocks import *

class UNetDDPM(nn.Module):
    def __init__(self, ch=64, ch_mult=(1, 2, 4, 8), num_res_blocks=1, attn_resolutions=[], z_channels=512, resolution=4800000, dropout=0.0, resamp_with_conv=True):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.z_channels = z_channels
        self.resolution = resolution
        self.t_emb_ch = 0
        self.resamp_with_conv = resamp_with_conv

        # Compute in_ch_mult, block_in, and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2**(self.num_resolutions - 1)

        # Downsampling layers
        self.down = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, t_emb_channels=self.t_emb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(LinearAttention(dim=block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != 0:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # Middle layers
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, t_emb_channels=self.t_emb_ch, dropout=dropout)
        self.mid.attn_1 = LinearAttention(dim=block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, t_emb_channels=self.t_emb_ch, dropout=dropout)

        # Upsampling layers
        self.up = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, t_emb_channels=self.t_emb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(LinearAttention(dim=block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != self.num_resolutions - 1:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.append(up)

        self.norm_out = nn.LayerNorm(block_in)
        self.conv_out = nn.Conv1d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z, t_emb=None):
        # List to store the outputs of each downsampling block for skip connections
        skip_connections = []

        # Downsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks):
                z = self.down[i_level].block[i_block](z, t_emb)
                if len(self.down[i_level].attn) > 0:
                    z = self.down[i_level].attn[i_block](z)
            skip_connections.append(z)
            if i_level != 0:
                z = self.down[i_level].downsample(z)

        # Middle layers
        z = self.mid.block_1(z, t_emb)
        z = self.mid.attn_1(z)
        z = self.mid.block_2(z, t_emb)

        # Upsampling
        for i_level in range(self.num_resolutions):
            z = z + skip_connections.pop()  # Skip connection
            for i_block in range(self.num_res_blocks):
                z = self.up[i_level].block[i_block](z, t_emb)
                if len(self.up[i_level].attn) > 0:
                    z = self.up[i_level].attn[i_block](z)
            if i_level != self.num_resolutions - 1:
                z = self.up[i_level].upsample(z)

        z = self.norm_out(z.transpose(1, 2)).transpose(1, 2)
        z = nn.SiLU()(z)
        z = self.conv_out(z)
        return z


import torch
import torch.nn as nn
import pytorch_lightning as pl

class DDPM(pl.LightningModule):
    def __init__(self, unet_model, T=1000, learning_rate=1e-4):
        super().__init__()
        self.unet_model = unet_model
        self.T = T
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()

    def forward(self, x, t):
        return self.unet_model(x, t)

    def training_step(self, batch, batch_idx):
        x_0 = batch  # Original data
        B, C, L = x_0.shape

        # Sample a random timestep for each example in the batch
        t = torch.randint(0, self.T, (B,), device=self.device).long()

        # Perform the forward diffusion process
        alpha_bar_t = 
        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar_t[:, None, None]) * x_0 + torch.sqrt(1 - alpha_bar_t[:, None, None]) * noise

        # Get the model's prediction for the noise
        eps_hat = self(x_t, t)

        # Compute the loss
        loss = self.loss_fn(eps_hat, noise)
        
        # Log loss
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    from ..variational_autoencoder.music_dataset import MusicDataset, collate_fn
    from ..variational_autoencoder.vae import VAE
    from torch.utils.data import DataLoader

    vae_model = VAE()
    unet_model = UNetDDPM(vae_model.encoder.z_channels, T=1000)
    ddpm_model = DDPM(unet_model)

    train_set = MusicDataset(root_dir=input("Data direrctory: "))

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1, collate_fn=collate_fn)
    val_loader = DataLoader(train_set, batch_size=1, num_workers=1, collate_fn=collate_fn)
    for batch in train_loader:
        print("Batch shape:", batch.shape)
        break

    trainer = pl.Trainer(max_epochs=100, precision='16-mixed')
    trainer.fit(ddpm_model, train_loader, val_loader)