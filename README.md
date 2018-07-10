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

The loss function for a VAE is the following:
$$L = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - KL(q_\phi(z|x)||p(z))$$

When using discrete latent variables we can't compute the gradient of the first term directly, but we can use REINFORCE to compute it:

$$\nabla \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z) \nabla_\phi \log q_\phi(z|x)]$$

In general we suppose that $q(z|x) = \prod_i q(z_i|x)$ and $p(z) = \prod_i p(z_i)$. Thus the KL can usually be computed in it's analytical:

$$KL(q_\phi(z|x)||p(z)) = \sum_i KL(q(z_i|x)||p(z_i))$$

When $p(z)$ is the uniform distribution then:

$$KL(q_\phi(z|x)||p(z)) = - H(q_\phi(z|x)) - \log \frac{1}{d}$$

where $H$ is the entropy. 

## 3. Gumbel-Softmax VAE (Not Implemented yet)
To train a VAE with reinforce (20 latent variables with each 256 categories):

```bash
python train_gumbel_vae.py
```
