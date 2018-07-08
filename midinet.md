---
description: >-
  A Convolutional Generative Adversarial Network for Symbolic-domain Music
  Generation
---

# MidiNet

## Background

By the time of MidiNet\(03/2017\) there are many deep-learning based music generation models, including WaveNet and MelodyRNN. But at that time majority of all endeavor was using RNN and their variants. WaveNet was the only major player who use CNN. One advantage of training CNN vs RNN is that the former is faster and more easily parallelizable. 

Another major deep learning breakthrough during that period is GAN\(Generative Adversarial Network\). The use of GAN provides the needed creativity for music. MidiNet also use GAN architecture with CNN-like generator and discriminator. In the case of MidiNet the generator is to transform random noises into a 2-D scorelike representation, that “appears” to be from real MIDI. Meanwhile the discriminator takes this 2-D scorelike representation and predicts whether this is real or not.

## Method

### Symbolic Representation for Convolution

The input MIDI files is split by bars. For each bar, we use a $$h\times w$$ matrix where $$h$$ is the number of MIDI notes that we consider. In their implementation this value is set to be 128 which represents all the notes between C0 and G10. However in their training model, they shifts all the melodies to only 2 scales C4 to B5\(Although you still have all the representations for 128 notes\). They claim doing in this way they can more easily model collapse by detecting if any generated note is outside C4 and B5. Also they did not put another dimension for silence as their training data does not have any.

## Generator and Discriminator

For a normal GAN, the loss function is like the below one where you play a minimax game between Discriminator and Generator by alternatively optimize over them.

$$
\underset{G}{min}\underset{D}{max} V(D,G)=E_{\mathbf{X}\sim P_{data}(X)}[log(D(x))]+E_{\mathbf{Z}\sim P_{z}(z)}[log(1-D(G(z)))]
$$

Other than this typical setting,   





