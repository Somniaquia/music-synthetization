if __name__ == "__main__":
    from models.latent_diffusion.music_dataset import MusicDataset
    from models.latent_diffusion.music_dataset import collate_fn
    from models.latent_diffusion.vae.vae import VAE
    from models.latent_diffusion.unet.unet import UNetDDPM, DDPM
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl

    vae_model = VAE()
    unet_model = UNetDDPM(vae_model.encoder.z_channels)
    ddim_model = DDPM(unet_model)

    train_set = MusicDataset(root_dir=input("Data direrctory: "))

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1, collate_fn=collate_fn)
    val_loader = DataLoader(train_set, batch_size=1, num_workers=1, collate_fn=collate_fn)
    for batch in train_loader:
        print("Batch shape:", batch.shape)
        break

    trainer = pl.Trainer(max_epochs=100, precision='16-mixed')
    trainer.fit(ddim_model, train_loader, val_loader)
