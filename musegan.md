---
description: >-
  Multi-track Sequential Generative Adversarial Networks for Symbolic Music
  Generation and Accompaniment
---

# MuseGAN

## Context

This is the follow up paper of MidiNet. The paper is originally published in 2017, and a later version of binary MuseGAN was published in 2018. It uses some ideas proposed in MidiNet including data representation and the use of GAN as basic network architecture. 

Here are some of the novelties of this paper

* Generate multi-track polyphonic music with temporal structure and multi-track interdependency.
* Extend the model to track-conditional generation.
* Intra-track and Inter-track objective measures for measuring real and generate music.

## Concepts

For data representation and the basic of using GANs, please refer to the previous page on [MidiNet](https://xihuanzeng.gitbook.io/deep-music-generation/~/edit/drafts/-LHdnX-x_EdoESjygn6p/midinet).

### Modeling the Multi-track Interdependency

#### Jamming Model

Suppose we have track 1...M. We then have M generators which works independently by generating its track based on its random vector $$z_{i}$$ .

For each generator, we also have its corresponding discriminator. The discriminator $$D_{i}$$ will try to discriminate $$G_i(z_i)$$ from real music track.

#### Composer Model

One single generator takes a random vector $$z$$ , and generate all tracks collectively with each channel of final output represents a track. We also only need one discriminator to tell if the input is real or fake.

#### Hybrid Model

This is hybrid of Jamming Model and Hybrid Model. We have M generators where each one takes two random vectors, which are the intra-track$$z_i$$and a shared inter-track $$z$$. There is only one discriminator D to make the final judgement. One advantage of Hybrid Model over Composer Model is that it can use different network architecture for each of the generator on different track which gives more flexibility.

###  Modeling the Temporal Structure

With the above model, although we have choice of measuring the inter-dependency of different tracks, we still have to generate the music bar by bar. That is to say, we need another modeling technique to account for the inter-dependency in the time dimension. Two models are designed for this.

###  Generation from Scratch

The first model uses two generators, namely the temporal generator $$G_{temp}$$ and the bar generator  $$G_{bar}$$. 

