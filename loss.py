import torch
from torch.nn import functional as F

import options
 
def KL_Loss(mu, logvar, beta):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return beta * KLD

def recon_Loss(recon_x, x, weight):
    # BCE = torch.nn.BCELoss(reduction="sum")
    batch_size = recon_x.shape[0]

    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / batch_size

    return weight * recon_loss

def reg_loss_sign(latent_code, attribute, factor=1.0):
   # Calculate both latent code and attribute distance matrices
   latent_code = latent_code.view(-1, 1).repeat(1, latent_code.shape[0])
   latent_code_dist_mat = (latent_code - latent_code.tranpose(1, 0)).view(-1, 1)

   attribute = attribute.view(-1, 1).repeat(1, attribute.shape[0])
   attribute_dist_mat = (attribute - attribute.transpose(1, 0)).view(-1, 1)

   # Compute reg loss
   loss_fn = torch.nn.L1Loss()
   latent_code_tanh = torch.tanh(latent_code_dist_mat * factor)
   attribute_sign = torch.sign(attribute_dist_mat)
   sign_loss = loss_fn(latent_code_tanh, attribute_sign.float())

   return sign_loss
   

def reg_Loss(latent_code, features, gamma=1.0, factor=1.0):
    AR_loss = 0

    for dim in range(features.shape[1]):
        x = latent_code[:, dim]
        features_dim = features[:,  dim]
        AR_loss += reg_loss_sign(x, features_dim, factor)

    return gamma * AR_loss