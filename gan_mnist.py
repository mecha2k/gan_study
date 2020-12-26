import torch
import torch.nn as nn
from IPython.display import Image

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

import time
from multiprocessing import freeze_support

latent_dim = 100


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(input_dim, output_dim, normalize=True):
            layers = [nn.Linear(input_dim, output_dim)]
            if normalize:
                layers.append(nn.BatchNorm1d(output_dim, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 1 * 28 * 28),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1 * 28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        flattened = img.view(img.size(0), -1)
        output = self.model(flattened)

        return output


def main():
    transforms_train = transforms.Compose(
        [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )

    train_dataset = datasets.MNIST(
        root="./dataset", train=True, download=True, transform=transforms_train
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4
    )

    generator = Generator()
    discriminator = Discriminator()

    generator.cuda()
    discriminator.cuda()

    adversarial_loss = nn.BCELoss()
    adversarial_loss.cuda()

    lr = 0.0002

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    import time

    n_epochs = 200
    sample_interval = 2000
    start_time = time.time()

    d_loss, g_loss = 0, 0
    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            real = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(1.0)
            fake = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(0.0)

            real_imgs = imgs.cuda()
            optimizer_G.zero_grad()
            z = torch.normal(mean=0, std=1, size=(imgs.shape[0], latent_dim)).cuda()
            generated_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(generated_imgs), real)
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), real)
            fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            done = epoch * len(dataloader) + i
            if done % sample_interval == 0:
                save_image(generated_imgs.data[:25], f"{done}.png", nrow=5, normalize=True)

        print(
            f"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]"
        )
        print(f"[Elapsed time: {time.time() - start_time:.2f}s]")

    Image("92000.png")


if __name__ == "__main__":
    freeze_support()
    main()