#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import sys
import wandb
import argparse
from torchsummary import summary

from loss import *
from model import *
from dataloader import *
from utils import *
import options

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main_train(args):
    # Train Loader
    if not os.path.exists(options.FEATURES_DF_PKL_FILEPATH):
        raise FileNotFoundError("Run preprocessing script to generate features dataframe first.")
    with open(options.FEATURES_DF_PKL_FILEPATH, 'rb') as file:
        features_df = pickle.load(file)
    train_loader = get_dataloader(options.IMG_PROCESSED_DIRPATH, options.IMGLIST_PROCESSED_FILEPATH, features_df)

    # Model
    model = AttriVAE(image_channels=options.IMG_CHANNELS, hidden_dim=options.HIDDEN_DIM, latent_dim=options.LATENT_DIM, encoder_channels=options.ENCODER_CHANNELS, decoder_channels=options.DECODER_CHANNELS).to(DEVICE)
    model.apply(initialize_weights)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.opt_lr)

    if args.summary:
        summary(model, (3, 128, 128))
        sys.exit(0)

    # Wandb Init
    wandb_run_name = args.custom_run_name if args.custom_run_name else f"[{get_time()}]-AttriVAE-{args.opt_lr}" 
    if args.custom_run_name:
        wandb_run_name = args.custom_run_name
    wandb.init(project="BirdMerge", entity="mochaminds", name=wandb_run_name)

    # Begin Training
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(args.num_epochs):
            train_loss = _train(epoch, model, train_loader, optimizer, do_AR_loss=args.do_AR_loss, do_L1_reg=args.do_L1_reg)
            wandb.log({"Train Loss": train_loss, "epoch": epoch + 1})
        
    # Save Model State
    torch.save(model.state_dict(), f'[{get_time()}]-STATE-AttriVAE.pt')

def _train(epoch_num, model, train_loader, optimizer, do_AR_loss, do_L1_reg):
    train_loss = 0
    for batch_idx, (data, features) in enumerate(train_loader):
        data, features = data.to(DEVICE), features.to(DEVICE)

        recon_batch, mu, logvar, z_tilde, z_dist, prior_dist = model(data)

        # Calculating loss
        recon_loss = recon_Loss(recon_x=recon_batch, x=data, weight=options.RECON_WEIGHT)
        kl_loss = KL_Loss(z_dist, prior_dist, options.BETA)
        loss = recon_loss + kl_loss
        print(f"Recon loss: {recon_loss.item()}; KL loss: {kl_loss.item()}; ", end="")

        if do_AR_loss: 
            attr_reg_loss = reg_Loss(z_tilde, features, gamma = options.GAMMA, factor = options.AR_FACTOR)
            loss += attr_reg_loss
            print(f"AR loss: {attr_reg_loss.item()}; ", end="")

        if do_L1_reg:
            # L1 Regularization
            l1_crit = nn.L1Loss(reduction="sum")
            weight_reg_loss = 0
            for param in model.parameters():
                weight_reg_loss += l1_crit(param, target=torch.zeros_like(param))
            loss += options.L1_REG_FACTOR * weight_reg_loss
            print(f"Reg loss: {weight_reg_loss.item()}; ", end="")
        print()

        # Update Weights
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss

    train_loss /= len(train_loader)
    print(f"E{str(epoch_num + 1).zfill(4)} || Train loss: {train_loss}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=options.NUM_EPOCHS, help="Number of epochs for training")
    parser.add_argument("--opt_lr", type=float, default=options.OPT_LR, help="Learning rate for optimizer")
    parser.add_argument("--summary", action="store_true", default=False, help="Flag to enable summary")
    parser.add_argument("--custom_run_name", type=str, default=None, help="Custom Wandb run name")
    parser.add_argument("--do_AR_loss", type=bool, default=options.DO_AR_LOSS, help="Flag to enable AR loss")
    parser.add_argument("--do_L1_reg", type=bool, default=options.DO_L1_REG, help="Flag to enable L1 regularization")
    args = parser.parse_args()

    main_train(args)
