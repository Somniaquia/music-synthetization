from functools import partial

import os
import glob

def get_latest_checkpoint(log_dir='lightning_logs'):
    version_dirs = glob.glob(os.path.join(log_dir, 'version_*'))

    if not version_dirs:
        raise FileNotFoundError("No version directories found in lightning_logs.")

    latest_version_dir = sorted(version_dirs, key=lambda x: int(x.split('_')[-1]))[-1]
    checkpoint_files = glob.glob(os.path.join(latest_version_dir, 'checkpoints', '*.ckpt'))

    if not checkpoint_files:
        return None
    
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)

    return latest_checkpoint

if __name__ == "__main__":
    option = input("Model to train: ")

    if option == "vae":
        from models.latent_diffusion.vae.vae import VAE
        from torch.utils.data import DataLoader
        from models.latent_diffusion.music_dataset import MusicDataset, collate_fn
        import pytorch_lightning as pl

        model = VAE(num_res_blocks=2, resolution=48000)
        train_set = MusicDataset(root_dir=input("Dataset path: "))

        collate_fn = partial(collate_fn, max_length=48000)

        train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=16, collate_fn=collate_fn)
        val_loader = DataLoader(train_set, batch_size=1, num_workers=16, collate_fn=collate_fn)

        for batch in train_loader:
            print("Batch shape:", batch.shape)
            break

        trainer = pl.Trainer(max_epochs=100, precision='32-true', log_every_n_steps=48)
        trainer.fit(model, train_loader, val_loader)

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

        trainer = pl.Trainer(max_epochs=100, precision='32-true', log_every_n_steps=48)
        trainer.fit(ddim_model, train_loader, val_loader)