import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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
        self.initial = nn.Parameter(torch.randn(1, 512, 4, 4))  # Starting from a learned constant
        self.mapping_network = MappingNetwork(nz, 512)
        self.style_blocks = nn.ModuleList([
            AdaIN(512, 512),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Upsample to 8x8
            AdaIN(512, 512),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Upsample to 16x16
            AdaIN(512, 256),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Upsample to 32x32
            AdaIN(512, 128),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Upsample to 64x64
            AdaIN(512, 64),
            nn.Conv2d(64, num_channels, 3, padding=1),
            nn.Tanh()
        ])  

    def forward(self, z):
        w = self.mapping_network(z)
        x = self.initial.repeat(z.size(0), 1, 1, 1)  # Repeat for batch size
        for layer in self.style_blocks:
            if isinstance(layer, AdaIN):
                x = layer(x, w)
            else:
                x = layer(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),  # Output size: 64 x 32 x 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # Output size: 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # Output size: 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # Output size: 512 x 4 x 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),  # Output size: 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)
    
def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nz = 100  # Size of z latent vector (i.e., size of generator input)
    lr = 0.9
    beta1 = 0.5
    batch_size = 64

    # Create the generator and discriminator
    netG = StyleGenerator(nz).to(device)
    netD = Discriminator().to(device)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Loss function
    criterion = nn.BCELoss()
    # Transformations
    transform = transforms.Compose([
        transforms.Resize(64),  # Ensuring the images are resized to 64x64
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Training code continues...
    num_epochs = 10
    # Load dataset
    dataset = dset.CIFAR10(root='./data', download=True, transform=transform)

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            # Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), 1, dtype=torch.float, device=device)

            print("Input size to discriminator:", real_cpu.size())
            output = netD(real_cpu)

            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(b_size, nz, device=device)
            fake = netG(noise)
            label.fill_(0)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimizerD.step()

            # Update Generator: maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(1)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD_real + errD_fake:.4f} Loss_G: {errG:.4f}')

            if (i % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(torch.randn(64, nz, device=device)).detach().cpu()
                save_image(fake.data, f'output/fake_samples_epoch_{epoch}_{i}.png', normalize=True)

if __name__ == '__main__':
    main()