# thais example largely guided by 
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

# first shot at a CVAE
class vae_cqt(nn.Module):

    # constructor
    def __init__(self) :
        super(vae_cqt, self).__init__()

        # encoder - let's try two layers
        
        # INPUT: 252 CQT bins
        self.encoding_layers = nn.Sequential(
            nn.Linear(252, 128),
            nn.ReLU(),
            nn.Linear(128, 64, bias=False),
            nn.ReLU(),
            #nn.Linear(128, 64, bias=False),
            #nn.ReLU()
        )

        self.mu_encoding = nn.Linear(64, 16, bias=False)  # mu layer
        self.var_encoding = nn.Linear(64, 16, bias=False)  # logvariance layer

        # decoder - symmetric architecture

        # INPUT: latent space
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 252, bias=False),
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
    #mse = F.mse_loss(x, x_hat)
    #kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #return (mse + beta*kld) / x.size(0)
    numerator = torch.sum((x - x_hat).pow(2))
    denominator = torch.sum(x.pow(2))
    sc = torch.sqrt(numerator / denominator)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (sc + beta*kld) / x.size(0)

# main function for training
if __name__ == '__main__' :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    # i should probably normalize the data....
    timbre_data = np.load('data/timbre_data_cqt.npy').T
    timbre_data = timbre_data / np.max(timbre_data, axis=1).reshape(timbre_data.shape[0], 1)
    timbre_data = torch.from_numpy(timbre_data).float()
    pitch_data = np.load('data/pitch_data_cqt.npy') # this will actually be used later, so we don't really need it here

    vae = vae_cqt()
    optimizer = torch.optim.Adam(vae.parameters()) # use default learning rate and other params
    num_epochs = 20
    batch_size = 10
    beta = 0.0005 # beta-vae
    beta_update = beta / 10
    bbeta = beta_update

    for epoch in range(num_epochs) :
        losses = []
        for i in range(pitch_data.shape[0] // batch_size) :
            X = timbre_data[i*batch_size:(i+1)*batch_size, :]
            x, x_hat, mu, logvar = vae(X)
            loss = loss_function(x, x_hat, mu, logvar, bbeta) # phasing in beta
            #loss = loss_function(x, x_hat, mu, logvar, beta)
            losses.append(loss.detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(str(epoch) + ' ' + str(np.mean(np.array(losses))))
        bbeta = min(beta, bbeta + beta_update)
    # save model here
    torch.save(vae.state_dict(), 'vae_cqt_model_params.pytorch')

