# Discrete VAE

This code requires version 0.4.0 of pytorch.
This code trains a VAE on BinaryMNIST based on https://github.com/pytorch/examples/tree/master/vae

## 1. VAE
To train a classic VAE with continuous latent variables:

```bash
python train_vae.py
```

## 2. REINFORCE VAE
To train a VAE with reinforce (20 latent variables with each 256 categories):

```bash
python train_reinforce_vae.py
```

## 3. Gumbel-Softmax VAE (Not Implemented yet)
To train a VAE with reinforce (20 latent variables with each 256 categories):

```bash
python train_gumbel-softmax_vae.py
```
