Repository for Prescribed Generative Adversarial Networks at https://arxiv.org/pdf/1910.04302.pdf

Abstract: Generative adversarial networks (GANs) are a powerful approach to unsupervised learning. They have achieved state-of-the-art performance in the image domain. However, GANs are limited in two ways. They often learn distributions
with low support—a phenomenon known as mode collapse—and they do not guarantee the existence of a probability density, which makes evaluating generalization using predictive log-likelihood impossible. In this paper, we develop
the prescribed GAN (PresGAN) to address these shortcomings. PresGANs add noise to the output of a density network and optimize an entropy-regularized adversarial loss. The added noise renders tractable approximations of the predictive log-likelihood and stabilizes the training procedure. The entropy regularizer encourages PresGANs to capture all the modes of the data distribution. Fitting PresGANs involves computing the intractable gradients of
the entropy regularization term; PresGANs sidestep this intractability using unbiased stochastic estimates. We evaluate PresGANs on several datasets and found they mitigate mode collapse and generate samples with high perceptual quality. We further found that PresGANs reduce the gap in performance in terms of predictive log-likelihood between traditional GANs and variational
auto-encoders (VAEs).

This repository contains a `pytorch` implementation for PresGANs. 

These GANs have

* High sample quality
* High sample diversity
* Tractable predictive log-likelihood

They use an entropy regularization term to overcome mode collapse, which is common with traditional GANs. 

![PresGAN loss](https://imgur.com/DYIewZe.png)

Example usage

`$ python main.py --dataset mnist --model presgan`

To cite please use the following bibtex

```
@article{dieng2019prescribed,
  title={Prescribed Generative Adversarial Networks},
  author={Dieng, Adji B and Ruiz, Francisco JR and Blei, David M and Titsias, Michalis K},
  journal={arXiv preprint arXiv:1910.04302},
  year={2019}
}
```
