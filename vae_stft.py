# this example largely guided by 
#       https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/
#       and
#       https://github.com/timbmg/VAE-CVAE-MNIST
# architecture design inspired by CANNe autoencoder https://arxiv.org/abs/2004.13172

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

class vae_stft(nn.Module):

    # constructor
    def __init__(self) :
        super(vae_stft, self).__init__()

        # encoder - let's try three layers
        
        # INPUT: 1025 FFT bins (positive frequencies fo 2048 bins)
        self.encoding_layers = nn.Sequential(
            nn.Linear(1025, 256),
            nn.ReLU(),
            nn.Linear(256, 64, bias=False),
            nn.ReLU(),
        )

        self.mu_encoding = nn.Linear(64, 16, bias=False)  # mu layer
        self.var_encoding = nn.Linear(64, 16, bias=False)  # logvariance layer

        # decoder - symmetric architecture

        # INPUT: latent space + f0
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1025, bias=False),
            nn.ReLU() # necessary? we do want only >= 0 outputs though
        )
    # end constructor

    def encode(self, x) :
        tmp = self.encoding_layers(x)
        return self.mu_encoding(tmp), self.var_encoding(tmp)

    def decode(self, z) :
        return self.decoder(z)

    # standard VAE reparameterization trick to make backprop work with the sampling operation
    def reparam_trick(self, mu, logvar) :
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    # forward pass, return all params relevant to the loss function
    def forward(self, x) :
        mu, logvar = self.encode(x)
        z = self.reparam_trick(mu, logvar)
        x_hat = self.decode(z)
        return x, x_hat, mu, logvar
# end class

# VAE loss function
def loss_function(x, x_hat, mu, logvar, beta) :
    numerator = torch.sum((x - x_hat).pow(2))
    denominator = torch.sum(x.pow(2))
    sc = torch.sqrt(numerator / denominator) # spectral convergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (sc + beta*kld) / x.size(0)

# main function for training
if __name__ == '__main__' :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    timbre_data = np.load('data/timbre_data.npy').T
    print(timbre_data.shape)
    # for some reason there are some totally empty rows
    good_rows = np.max(timbre_data, axis=1) > 0
    print(good_rows.shape)
    timbre_data = timbre_data[good_rows,:]
    print(timbre_data.shape)
    # normalize
    timbre_data = timbre_data / np.max(timbre_data, axis=1).reshape(timbre_data.shape[0], 1)
    timbre_data = torch.from_numpy(timbre_data).float()
    pitch_data = np.load('data/pitch_data.npy') # this will actually be used later, so we don't really need it here
    pitch_data = pitch_data[good_rows]

    vae = vae_stft()
    optimizer = torch.optim.Adam(vae.parameters()) # use default learning rate and other params
    num_epochs = 20
    batch_size = 10
    beta = 0.001 # beta-vae
    beta_update = beta / 10
    bbeta = beta_update

    for epoch in range(num_epochs) :
        losses = []
        for i in range(pitch_data.shape[0] // batch_size) :
            X = timbre_data[i*batch_size:(i+1)*batch_size, :]
            x, x_hat, mu, logvar = vae(X)
            loss = loss_function(x, x_hat, mu, logvar, bbeta) # phasing in beta
            losses.append(loss.detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(str(epoch) + ' ' + str(np.mean(np.array(losses))))
        bbeta = min(beta, bbeta + beta_update)
    # save model here
    torch.save(vae.state_dict(), 'vae_stft_model_params.pytorch')
