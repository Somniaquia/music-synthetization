# Code taken from https://github.com/CompVis/stable-diffusion.git
# for personal learning purposes

from einops import rearrange
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

torch.set_float32_matmul_precision('medium')

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models (https://arxiv.org/abs/2006.11239):
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb

def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='linear')  # Changed to 'linear' for smoother results
        if self.with_conv:
            x = self.conv(x)
        return x
    
class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = F.avg_pool1d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, t_emb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        out_channels = self.out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = nn.LayerNorm(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if t_emb_channels > 0:
            self.t_emb_proj = nn.Linear(t_emb_channels, out_channels)
        
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                # The NiN (Network-in-Network) shortcut - https://arxiv.org/abs/1312.4400
                self.nin_shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, t_emb):
        h = x

        h = self.norm1(h.transpose(1, 2)).transpose(1, 2)
        h = nn.SiLU()(h)
        h = self.conv1(h)

        # Project t_emb to the same dimensionality as the output of the first convolutional layer, add to the result
        if t_emb is not None:
            # The original network processed 4D image tensors (b, h, w, c) but in case of 2D audio tensors (b, s) dimensionality augmentation is unnecessary
            # h = h + self.t_emb_proj(nn.SiLU(t_emb))[:,:,None,None]
            h = h + self.t_emb_proj(nn.SiLU()(t_emb))[:, :, None]

        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        h = nn.SiLU()(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        # Implement the defining skip connection of a ResnetBlock
        return x+h

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, l = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) l -> qkv b heads c l', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c l -> b (heads c) l', heads=self.heads, l=l)
        return self.to_out(out)

# Gaussian Distribution with a diagonal covariance matrix
class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        # A tensor whose first half along the first dimension represents means, and the second half represents log variances of a Gaussian distribution.
        self.parameters = parameters
        # Means and Log Variances of the Gaussian distribution
        self.mean, self.logvar = torch.chunk(parameters, chunks=2, dim=1)
        # Clamp logvars in order to avoid numerical instability
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        # A boolean flag to determine whether the distribution should ignore the stochasticity and just use the mean during sampling.
        self.deterministic = deterministic
        # Compute standard deviation
        self.std = torch.exp(0.5 * self.logvar)
        # Compute variance
        self.var = torch.exp(self.logvar)

        # If deterministic == True, variance and std are set to zero tensors
        # This results in subsequent sampling yielding the means of the distributions
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    # Sample from the random distribution - or return the mean if deterministic == True
    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x
    
    # Calculates KL Divergence from an another Gaussian distribution
    # If the other distribution is not provided, this calculates KL Divergence with a normal distribution N(1, 0)
    def kl(self, other=None):
        if self.deterministic:
            return torch.tensor([0.], device=self.parameters.device)
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1])
            else:
                return 0.5 * torch.sum(torch.pow(self.mean - other.mean, 2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar, dim=[1])
            
    # Calculate the negative log-likelihood of a 'sample' under the distribution.
    # Common for loss functions of generative models
    def nll(self, sample, dims=[1]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)
    
    # Returns the mode (most probable value) of the distribution, which is the mean in this case
    def mode(self):
        return self.mean
    
class Encoder(nn.Module):
    def __init__(self, ch=128, out_ch=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
                 attn_resolutions=[], dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=256, z_channels=16, give_pre_end=False, tanh_out=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.t_emb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.resolution = resolution
        self.in_channels = in_channels

        # Downsampling layers
        self.conv_in = nn.Conv1d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = self.resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, t_emb_channels=self.t_emb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(LinearAttention(dim=block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # Middle Layers
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, t_emb_channels=self.t_emb_ch, dropout=dropout)


    def forward(self, x, t_emb=None):
        # Initial convolution
        h = self.conv_in(x)

        # Downsampling layers
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, t_emb)

                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # Middle layers
        h = self.mid.block_1(h, t_emb)

        return h
    
class Decoder(nn.Module):
    def __init__(self, ch=128, out_ch=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2, 
                 attn_resolutions=[], dropout=0.0, resamp_with_conv=True, in_channels=3, 
                 resolution=256, z_channels=16, give_pre_end=False, tanh_out=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # Compute in_ch_mult, block_in, and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2**(self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res)
        print(f"Working with z of shape {self.z_shape} dimensions.")

        self.conv_in = nn.Conv1d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, t_emb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = LinearAttention(dim=block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, t_emb_channels=self.temb_ch, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, t_emb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(LinearAttention(dim=block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = nn.LayerNorm(block_in)
        self.conv_out = nn.Conv1d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
         

    def forward(self, z):
        self.last_z_shape = z.shape
        t_emb = None

        h = self.conv_in(z)
        h = self.mid.block_1(h, t_emb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, t_emb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, t_emb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h
        
        h = self.norm_out(h.transpose(1, 2)).transpose(1, 2)
        h = nn.SiLU()(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class TimeDomainLoss(nn.Module):
    def __init__(self):
        super(TimeDomainLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, y_pred, y_true):
        return self.l1_loss(y_pred, y_true)

class FrequencyDomainLoss(nn.Module):
    def __init__(self):
        super(FrequencyDomainLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, y_pred, y_true):
        # Compute the STFT of y_pred and y_true
        stft_pred = torch.stft(y_pred, n_fft=1024, hop_length=256, win_length=1024)
        stft_true = torch.stft(y_true, n_fft=1024, hop_length=256, win_length=1024)

        # Compute the magnitude of the STFT
        mag_pred = torch.sqrt(stft_pred.pow(2).sum(-1))
        mag_true = torch.sqrt(stft_true.pow(2).sum(-1))

        # Compute the L1 loss between the magnitudes
        loss = self.l1_loss(mag_pred, mag_true)
        return loss
    
class CombinedAudioLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedAudioLoss, self).__init__()
        self.time_domain_loss = TimeDomainLoss()
        self.frequency_domain_loss = FrequencyDomainLoss()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        time_loss = self.time_domain_loss(y_pred, y_true)
        freq_loss = self.frequency_domain_loss(y_pred, y_true)
        loss = (1 - self.alpha) * time_loss + self.alpha * freq_loss
        return loss


class VAE(pl.LightningModule):
    def __init__(self,
                 embed_dim=64,
                 ckpt_path=None,
                 ignore_keys=[],
                 monitor=None,
                 learning_rate=1e-3
                 ):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.loss = CombinedAudioLoss(alpha=0.5)
        self.lossconfig = {}
        self.quant_conv = nn.Conv1d(2*16, 2*embed_dim, 1)
        self.post_quant_conv = nn.Conv1d(embed_dim, 16, 1)
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location='cpu')['state_dict']
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print('Deleting key {} from state_dict'.format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f'Reconstructed from {path}')

    # Return the 'latent vector (posterior)' of a VAE, which is a multivariate Gaussian Distribution
    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    # Decode, try to reconstruct a sample from the latent vector
    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x, sample_posterior):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        # Return the reconstruction and the posterior
        return dec, posterior
    
    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        return x.float()

    # A mandatory function to define what optimizers to use in pytorch lightning.
    # Returns the optimizers and optionally rate schedulers used in training
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        # Calculate the combined audio loss
        loss = self.combined_audio_loss(reconstructions, inputs)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, _ = self(inputs)

        # Calculate the combined audio loss
        val_loss = self.combined_audio_loss(reconstructions, inputs)
        self.log('val_loss', val_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return val_loss
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
if __name__ == "__main__":
    from music_dataset import MusicDataset
    from torch.utils.data import DataLoader

    model = VAE()

    train_set = MusicDataset(root_dir='data/raw')
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=16)
    val_loader = DataLoader(train_set, batch_size=16, num_workers=16)

    trainer = pl.Trainer(max_epochs=100, precision='16-mixed')
    trainer.fit(model, train_loader, val_loader)