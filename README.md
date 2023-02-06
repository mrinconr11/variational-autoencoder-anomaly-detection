# variational-autoencoder-anomaly-detection
This repository contains a convolutional variational autoencoder (VAE) for annomaly detection in natural surface waters. 

The variational autoencoder is used to encode excitation-emission matrices (EEM). The encoded EEMs are used as input to a K-means model that classify the water samples between contaminated or nor-contaminated. 

The code also includes the continuous distribution of the EEMs. This graphics are made by interpolating across the latent space and making predictions with the decoder part of the VAE.

The models proposed resulted in a Rand Score 0.935, F Score of 0.97, and a Brier score of 0.02.
