import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import time


class MappingNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim, num_layers=8):
        super(MappingNetwork, self).__init__()
        layers = [nn.Linear(input_dim, feature_dim)]
        for _ in range(1, num_layers):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(feature_dim, feature_dim))  # Ensure feature_dim consistency
        self.model = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.model(z)

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super(AdaIN, self).__init__()
        self.style_scale_transform = nn.Linear(style_dim, num_features)
        self.style_shift_transform = nn.Linear(style_dim, num_features)
    
    def forward(self, content, style):
        scale = self.style_scale_transform(style).unsqueeze(2).unsqueeze(3)
        shift = self.style_shift_transform(style).unsqueeze(2).unsqueeze(3)
        normalized_content = nn.functional.instance_norm(content)
        stylized_content = scale * normalized_content + shift
        return stylized_content

class StyleGenerator(nn.Module):
    def __init__(self, nz, num_channels=3):
        super(StyleGenerator, self).__init__()
        self.mapping_network = MappingNetwork(nz, 512)
        self.initial = nn.Parameter(torch.randn(1, 512, 4, 4))     # Randomize initial tensor
        self.style_blocks  = nn.ModuleList([

            # Deconvolve tensor
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),  # 8x8
            nn.SiLU(inplace=True),
            AdaIN(512, 512), 
            nn.Conv2d(512, 512, 3, stride=1, padding=1),  # 8x8
            nn.SiLU(inplace=True),
            AdaIN(512, 512),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),  # 8x8
            nn.SiLU(inplace=True),
            AdaIN(512, 512),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 16x16
            nn.SiLU(inplace=True),
            AdaIN(512, 256),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),  # 16x16
            nn.SiLU(inplace=True),
            AdaIN(512, 256),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),  # 16x16
            nn.SiLU(inplace=True),
            AdaIN(512, 256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 32x32
            nn.SiLU(inplace=True),
            AdaIN(512, 128),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),  # 32x32
            nn.SiLU(inplace=True),
            AdaIN(512, 128),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),  # 32x32
            nn.SiLU(inplace=True),
            AdaIN(512, 128),  
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 64x64
            nn.SiLU(inplace=True),
            AdaIN(512, 64),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),  # 64x64
            nn.SiLU(inplace=True),
            AdaIN(512, 64),
            nn.Conv2d(64, num_channels, 3, stride=1, padding=1),  # 64x64
            nn.Tanh()
        ])

    def forward(self, z):
        style_codes = self.mapping_network(z)
        x = self.initial.expand(z.size(0), -1, -1, -1)
        for style_block in self.style_blocks:
            if isinstance(style_block, AdaIN):
                x = style_block(x, style_codes)
            else:
                x = style_block(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    nz = 100 # Size of z latent vector (i.e., size of generator input)
    lrD = 0.00015
    lrG = 0.00015
    beta1 = 0.5
    batch_size = 32         

    criterion = nn.BCELoss()

    # Create the generator and discriminator
    netG = StyleGenerator(nz).to(device)
    netD = Discriminator().to(device)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr = lrD, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr = lrG, betas=(beta1, 0.999))

    # optimizerD = optim.Adadelta(netD.parameters(), lr = 0.001, rho=0.1)
    # optimizerG = optim.Adadelta(netG.parameters(), lr = 0.001, rho=0.)

    # Transformations
    transform = transforms.Compose([
        transforms.Resize(64),  # Ensuring the images are resized to 64x64
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    num_epochs = 200
    dataset = dset.CIFAR10(root='./data', download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    total_time = 0.0
    time_per_epoch = []

    start_time = None
    end_time = None
    duration = None

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Processing epoch #{epoch}")
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real_cpu = data[0].to(device)
            noise = torch.randn(batch_size, nz, device=device)

            # Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            output_real = netD(real_cpu).view(-1)
            # Use label smoothing for real labels
            label_real = torch.full(tuple(output_real.size()), 0.9, device=device, dtype=torch.float)
            errD_real = criterion(output_real, label_real)
            fake = netG(noise)
            output_fake = netD(fake.detach()).view(-1)
            # Use label smoothing for fake labels
            label_fake = torch.full(tuple(output_fake.size()), 0.1, device=device, dtype=torch.float)
            errD_fake = criterion(output_fake, label_fake)
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()


            # Update Generator: maximize log(D(G(z)))
            netG.zero_grad()
            output = netD(fake).view(-1)
            # Use label smoothing for real labels
            label_real = torch.full(tuple(output.size()), 0.9, device=device, dtype=torch.float)
            errG = criterion(output, label_real)
            errG.backward()
            optimizerG.step()


            if i % 250 == 0 or i == len(dataloader)-1:
                print(f'\t[{epoch+1}/{num_epochs}][{i+1}/{len(dataloader)}] Loss_D: {errD:.4f} Loss_G: {errG:.4f}\n')
                fake = netG(torch.randn(64, nz, device=device)).detach().cpu()
                save_image(fake.data, f'output/fake_samples_styleGAN.png', normalize=True)

        end_time = time.time()
        duration = end_time - start_time
        time_per_epoch.append(duration)
        total_time += duration
        avg_time = total_time/len(time_per_epoch)

        print(f"Completed epoch #{epoch}\tTime: {round(duration,2)} seconds\nAvg s/epoch: {round(avg_time,2)}\tTotal time: {round(total_time,2)}")

if __name__ == '__main__':
    main()
