import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tf
# from torch.autograd.variable import Variable
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader as DL
import argparse
import wandb
import matplotlib.pyplot as plt
import gc
import os

class generator(nn.Module):
    def __init__(self, input_size, output_size):
        hidden_size = output_size
        super(generator, self).__init__()
        self.input_size = input_size
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
        pass
        
    def forward(self, x):
        x = self.activation(self.map1(x))
        x = self.activation(self.map2(x))
        x = self.activation(self.map3(x))
        x = nn.Tanh()(x)
        return x
    pass


class discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.sigmoid_layer = nn.Sigmoid()
        self.activation = nn.LeakyReLU(0.01)
        #initialize using xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
        pass
    
    def forward(self, x):
        x = self.activation(self.map1(x))
        x = self.activation(self.map2(x))
        x = self.activation(self.map3(x))
        x = self.sigmoid_layer(x)
        return x
    pass



def train_gan(Gen, Dis, Gen_opt, Dis_opt, criterion, data_loader, args):
    Gen.train()
    Dis.train()
    Gen_hist = [] 
    Dis_hist = []
    mem_alloc = []
    
    for epoch in range(args.epochs):

        for i, (real_data, _) in enumerate(data_loader):
            
            torch.cuda.empty_cache()
            gc.collect()
            
            real_data = real_data.view(-1, 784).to(args.device)
            real_label = torch.ones(args.batch_size, 1).to(args.device)
            fake_label = torch.zeros(args.batch_size, 1).to(args.device)
            
            # breakpoint()
            # Train the discriminator
            Dis_opt.zero_grad()
            real_output = Dis(real_data)
            real_loss = criterion(real_output, real_label)
            
            z = torch.randn(args.batch_size, Gen.input_size).to(args.device)
            fake_data = Gen(z)
            fake_output = Dis(fake_data)
            fake_loss = criterion(fake_output, fake_label)
            
            Dis_loss = real_loss + fake_loss
            Dis_loss.backward()
            Dis_hist.append(Dis_loss.item())
            Dis_opt.step()
            
            
            # Train the generator every k steps 
            # if i%args.k == 0:
            print(f"Epoch {epoch}/{args.epochs}, Step {i}/{len(data_loader)}, D Loss: {Dis_loss.item()}")
                # for _ in range(args.num_gen):
            Gen_opt.zero_grad()
            # z = torch.randn(args.batch_size, Gen.input_size).to(args.device)
            fake_data = Gen(z)
            fake_output = Dis(fake_data)
            Gen_loss = criterion(fake_output, real_label)
            Gen_loss.backward()
            Gen_hist.append(Gen_loss.item())
            Gen_opt.step()
                # print the generator loss over the discriminator loss
            print(f"Final Generator Loss {Gen_loss.item()}")
                
            if i% 500  == 0:
                fig, ax = plt.subplots(2, figsize=(10, 5))
                ax[0].plot(Dis_hist, label='Discriminator Loss', color='red')
                ax[1].plot(Gen_hist, label = 'Generator Loss', color='blue')
                fig.legend()
                fig.savefig(f'{args.lr}/{args.epochs}/losses_{i}.png')
                plt.close()
                # now visualize the generated images
                z = torch.randn(10, Gen.input_size).to(args.device)
                fake_data = Gen(z)
                fake_data = fake_data.view(-1, 28, 28).detach().cpu().numpy()
                fig, ax = plt.subplots(1, 10, figsize=(20, 2))
                for index in range(10):
                    ax[index].imshow(fake_data[index], cmap='gray')
                    ax[index].axis('off')
                fig.savefig(f'{args.lr}/{args.epochs}/images_{i}.png')
                plt.close()
                pass
        pass
    pass

def visualize(data_loader, indices, substring='real'):
    fig, ax = plt.subplots(1, 10, figsize=(20, 2))
    for i, idx in enumerate(indices):
        ax[i].imshow(data_loader.dataset.data[idx], cmap='gray')
        ax[i].axis('off')
    plt.savefig('images' + substring + '.png')
    plt.close()
    pass

if __name__ == '__main__':
    
    transform = tf.Compose([tf.ToTensor(), tf.Normalize((0.1307), (0.3081,))])
    train_ds = DL(MNIST(root="./data", train=True, download = True, transform= transform) ,batch_size=32)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for experiment tracking')
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--num_gen', type=int, default=1)
    args = parser.parse_args()
    
    
    #choose random 10 indices
    indices = np.random.choice(60000, 10)
    
    # visualize these images
    visualize(train_ds, indices)
    
    Gen = generator(1024, 784)
    Dis = discriminator(784, 256, 1)
    
    Gen.to(args.device)
    Dis.to(args.device)
    
    
    criterion = nn.BCELoss()
    Gen_opt = optim.Adam(Gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    Dis_opt = optim.Adam(Dis.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    if not os.path.exists(f'./{args.lr}'):
        os.makedirs(f'./{args.lr}')
        pass
    
    if not os.path.exists(f'./{args.lr}/{args.epochs}'):
        os.makedirs(f'./{args.lr}/{args.epochs}')
        pass
    
    train_gan(Gen, Dis, Gen_opt, Dis_opt, criterion, train_ds, args)
    
    # now make a gif or animation of how it changes suring the gan pricess

    
    # test_ds = DL(MNIST(root="./data", train=False, download = True, transform= transform) ,batch_size=35)
    pass


    # Define the model
    # Define the loss function
    # Define the optimizer
    # Train the model
    # Save the model
    # TestW