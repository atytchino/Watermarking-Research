import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils


# ---------------------------
# 1) Custom dataset to return index (optional)
#    This helps if you want a unique filename per image
# ---------------------------
class IndexedImageFolder(datasets.ImageFolder):
    """
    Subclass of ImageFolder that also returns the index of each sample.
    """

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return img, label, index


# ---------------------------
# 2) Define UNet-like Autoencoder
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.double_conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNetAutoencoder(nn.Module):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()

        # Down (encoder)
        self.down1 = DownBlock(3, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)

        # Pool
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Up (decoder)
        self.up1 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)

        # Final 1x1 -> 3 channels
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)  # (N, 64, 512, 512)
        x2 = self.pool(x1)

        x2 = self.down2(x2)  # (N, 128, 256, 256)
        x3 = self.pool(x2)

        x3 = self.down3(x3)  # (N, 256, 128, 128)
        x4 = self.pool(x3)

        x4 = self.down4(x4)  # (N, 512, 64, 64)

        # Decoder
        x = self.up1(x4, x3)  # (N, 256, 128, 128)
        x = self.up2(x, x2)  # (N, 128, 256, 256)
        x = self.up3(x, x1)  # (N, 64, 512, 512)

        x = self.final_conv(x)  # (N, 3, 512, 512)
        return x


# ---------------------------
# 3) Main code using nn.DataParallel
# ---------------------------
if __name__ == "__main__":
    # Settings
    data_dir = r"D:\Dropbox\UMA Augusta\PhD\Research Thesis\brain_tumor_mri_dataset\Training"
    save_dir = "upsized_experimental"
    os.makedirs(save_dir, exist_ok=True)

    batch_size = 2  # If you have 2 GPUs, each GPU will process 1 image/iteration
    num_workers = 2
    lr = 1e-6
    num_epochs = 3  # or however many you want if training

    # Check if GPUs are available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPUs. Device: {device}")

    # Dataset & DataLoader
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    dataset = IndexedImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # for training, shuffle is typical
        num_workers=num_workers,
        pin_memory=True
    )

    # Create model
    model = UNetAutoencoder()

    # Wrap with nn.DataParallel if multiple GPUs are available
    if n_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(n_gpus)))

    # Move model to device
    model = model.to(device)

    # Example: if you want to train, define loss & optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training Loop (optional)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, labels, indices) in enumerate(dataloader):
            images = images.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, images)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Save the upsampled images *immediately* if you want:
            # (Typically for training, you might do it occasionally or at the end.)
            with torch.no_grad():
                for i in range(outputs.size(0)):
                    out_img = outputs[i].cpu().clamp(0, 1)
                    idx = indices[i].item()
                    # create a filename using the sample index
                    fname = f"epoch{epoch + 1}_batch{batch_idx + 1}_idx{idx}.png"
                    fpath = os.path.join(save_dir, fname)
                    vutils.save_image(out_img, fpath, normalize=True, value_range=(0, 1))

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch + 1}/{num_epochs}] Avg Loss: {avg_loss:.4f}")

    print("Training complete!")
