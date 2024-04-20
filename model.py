import torch
import torch.nn as nn
from torchsummary import summary

def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
    # nn.init.xavier_uniform_(m.weight)
    nn.init.uniform_(m.weight, -0.08, 0.08)
    m.bias.data.fill_(0.01)
  elif isinstance(m, nn.BatchNorm2d):
    nn.init.constant_(m.weight.data, 1)
    nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm1d):
    nn.init.constant_(m.weight.data, 1)
    nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
    # nn.init.xavier_uniform_(m.weight.data)
    nn.init.uniform_(m.weight, -0.08, 0.08)
    nn.init.constant_(m.bias.data, 0)
     
class AttriVAE(nn.Module):
    def __init__(self, image_channels, hidden_dim, latent_dim, encoder_channels, decoder_channels):
        super(AttriVAE, self).__init__()
        self.image_channels = image_channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        
        self.encoder = nn.Sequential(nn.Conv2d(in_channels = image_channels, out_channels = self.encoder_channels[0], kernel_size=3, stride=2, padding=1),
                                     nn.BatchNorm2d(self.encoder_channels[0]),
                                     nn.LeakyReLU(),
                                     
                                     nn.Conv2d(in_channels = self.encoder_channels[0], out_channels = self.encoder_channels[1], kernel_size=3, stride=2, padding=1),
                                     nn.BatchNorm2d(self.encoder_channels[1]),
                                     nn.LeakyReLU(),
                                     
                                     nn.Conv2d(in_channels = self.encoder_channels[1], out_channels = self.encoder_channels[2], kernel_size=3, stride=2, padding=1),
                                     nn.BatchNorm2d(self.encoder_channels[2]),
                                     nn.LeakyReLU(),
                                     
                                     nn.Conv2d(in_channels = self.encoder_channels[2], out_channels = self.encoder_channels[3], kernel_size=3, stride=2, padding=1),
                                     nn.BatchNorm2d(self.encoder_channels[3]),
                                     nn.LeakyReLU(),
                                     
                                     nn.Conv2d(in_channels = self.encoder_channels[3], out_channels = self.encoder_channels[4], kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(self.encoder_channels[4]),
                                     nn.LeakyReLU(),)
        
        self.fc_encoder = nn.Sequential(nn.Flatten(), 
                                        nn.Linear(8 * 8 * self.encoder_channels[4], self.hidden_dim)) # @henry: Attri-VAE has intermediate layer here
        
        self.mean = nn.Linear(self.hidden_dim, self.latent_dim)
        self.logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        self.fc_decoder = nn.Sequential(nn.Linear(self.latent_dim, self.hidden_dim),
                                        nn.Linear(self.hidden_dim, 8 * 8 * self.encoder_channels[4]),)
        
        self.decoder = nn.Sequential(nn.Conv2d(in_channels = self.encoder_channels[4], out_channels=self.decoder_channels[0], kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(self.decoder_channels[0]),
                                     nn.LeakyReLU(),
                                     
                                     nn.ConvTranspose2d(in_channels = self.decoder_channels[0], out_channels=self.decoder_channels[0], kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(self.decoder_channels[0]),
                                     nn.LeakyReLU(),
                                     
                                     nn.ConvTranspose2d(in_channels = self.decoder_channels[0], out_channels=self.decoder_channels[1], kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(self.decoder_channels[1]),
                                     nn.LeakyReLU(),
                                     
                                     nn.ConvTranspose2d(in_channels = self.decoder_channels[1], out_channels=self.decoder_channels[2], kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(self.decoder_channels[2]),
                                     nn.LeakyReLU(),
                                     
                                     nn.ConvTranspose2d(in_channels = self.decoder_channels[2], out_channels=self.decoder_channels[3], kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(self.decoder_channels[3]),
                                     nn.LeakyReLU(),
                                     
                                     nn.Conv2d(in_channels = self.decoder_channels[3], out_channels=self.decoder_channels[4], kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(self.decoder_channels[4]),
                                     nn.LeakyReLU(),
                                     
                                     nn.Conv2d(in_channels = self.decoder_channels[4], out_channels=image_channels, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(image_channels),
                                     nn.LeakyReLU(),
                                     
                                     nn.Sigmoid()) # for [0, 1] to match ToTensor

        mlp_dims = [int(latent_dim / x) for x in (2, 4)]
        self.mlp = nn.Sequential(nn.Linear(latent_dim, mlp_dims[0]),
                                 nn.BatchNorm1d(mlp_dims[0]),
                                 nn.LeakyReLU(),
                                 nn.Linear(mlp_dims[0], mlp_dims[1]),
                                 nn.BatchNorm1d(mlp_dims[1]),
                                 nn.LeakyReLU(),
                                 nn.Linear(mlp_dims[1], 1),
                                 nn.Sigmoid())
        
    def reparametrize(self, mu, logvar):
        epsilon = torch.randn_like(logvar)
        z_sampled = mu + torch.sqrt(torch.exp(logvar)) * epsilon
        
        z_dist = torch.distributions.Normal(loc=mu, scale=torch.exp(logvar))
        z_tilde = z_dist.rsample()
        
        prior_dist = torch.distributions.Normal(loc=torch.zeros_like(z_dist.loc),scale=torch.ones_like(z_dist.scale))
        z_prior = prior_dist.sample()

        return z_sampled, z_tilde, z_prior, z_dist, prior_dist
    
    def forward(self, x):
        encoded = self.encoder(x)
        # print(f"Shape after encoder: {encoded.shape}")

        encoded = self.fc_encoder(encoded)
        # print(f"Shape after fc_encoder: {encoded.shape}")

        mu = self.mean(encoded)
        logvar = self.logvar(encoded)
        
        z_sampled, z_tilde, z_prior, z_dist, prior_dist = self.reparametrize(mu, logvar)
        # print(f"Shape of z_tilde: {z_tilde.shape}")

        decoded = self.fc_decoder(z_tilde) 
        # print(f"Shape after fc_decoder: {decoded.shape}")
        
        reshaped = decoded.view(decoded.size(0), self.encoder_channels[4], 8, 8) # @henry: double check view values
        decoded = self.decoder(reshaped)
        # print(f"Shape after decoder: {decoded.shape}")

        return decoded, mu, logvar, z_tilde, z_dist, prior_dist
        
if __name__ == "__main__":
    model = AttriVAE(3, 96, 64, (8, 16, 32, 64, 2), (64, 32, 16, 8, 4, 2))
    summary(model, (3, 128, 128))