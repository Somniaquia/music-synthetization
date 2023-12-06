from functools import partial
import librosa
from matplotlib import pyplot as plt

import torch
from data.spectogram import get_random_file_in_subfolders, plot_spectogram
from train import get_latest_checkpoint

if __name__ == "__main__":
    option = input("Model to evaluate: ")

    if option == "vae":
        from models.latent_diffusion.vae.vae import VAE
        from torch.utils.data import DataLoader
        from models.latent_diffusion.music_dataset import MusicDataset, collate_fn
        import pytorch_lightning as pl

        # path = get_random_file_in_subfolders("data\8kHz_8bit")
        # path = "C:\Somnia\Projects\music-synthetization\data\8kHz_8bit\ぬゆり\フラジール - GUMI _ Fragile - nulut.wav"
        path = input("Music Path: ")
        y, sr = librosa.load(path)

        processed_y = collate_fn([torch.tensor(y).unsqueeze(0)], max_length=48000).squeeze(0)
        
        model = VAE(num_res_blocks=2, resolution=48000)
        model.eval()

        model.load_state_dict(torch.load(get_latest_checkpoint())['state_dict'])
        y_tensor = torch.tensor(y).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            reconstructed_y = model(y_tensor).squeeze().numpy()

        collate_fn = partial(collate_fn, max_length=48000)
        
        plot_spectogram(y, sr)
        plot_spectogram(reconstructed_y, sr)


    elif option == 'unet':
        from models.latent_diffusion.vae.vae import VAE
        from models.latent_diffusion.unet.unet import UNetDDPM, DDPM
        from torch.utils.data import DataLoader
        from models.latent_diffusion.music_dataset import MusicDataset, collate_fn
        import pytorch_lightning as pl

        vae_model = VAE()
        unet_model = UNetDDPM(vae_model.encoder.z_channels)
        ddim_model = DDPM(unet_model)
        train_set = MusicDataset(root_dir=input("Data direrctory: "))

        collate_fn = partial(collate_fn, max_length=48000)

        train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1, collate_fn=collate_fn)
        val_loader = DataLoader(train_set, batch_size=1, num_workers=1, collate_fn=collate_fn)

        for batch in train_loader:
            print("Batch shape:", batch.shape)
            break

        trainer = pl.Trainer(
            max_epochs=100, precision='32-true', log_every_n_steps=48)
        trainer.fit(ddim_model, train_loader, val_loader)
