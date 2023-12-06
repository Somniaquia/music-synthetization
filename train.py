from functools import partial

if __name__ == "__main__":
    option = input("Model to train: ")

    if option == "vae":
        from models.latent_diffusion.vae.vae import VAE
        from torch.utils.data import DataLoader
        from models.latent_diffusion.music_dataset import MusicDataset, collate_fn
        import pytorch_lightning as pl

        model = VAE()
        train_set = MusicDataset(root_dir="data\8kHz_8bit")

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