---
description: A Generative Model for Raw Audio
---

# WaveNet

## Background

WaveNet\([https://arxiv.org/abs/1609.03499](https://arxiv.org/abs/1609.03499)\) is a neural network architecture published by DeepMind at Sep 2016. The work is inspired by generative models such as  Pixel CNN\([https://arxiv.org/abs/1606.05328](https://arxiv.org/abs/1606.05328)\) and Pixel RNN\([https://arxiv.org/abs/1601.06759](https://arxiv.org/abs/1601.06759)\) on generating images and text.

The model is mainly tackling the task of text-to-speech \(TTS\), which takes text as input and generate human-like speech. The novelty of WaveNet is that it uses a parameterized way to generate raw speech signal in comparison to the by-the-time state-of-the-art [concatenative TTS](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Es-YRKMAAAAJ&citation_for_view=Es-YRKMAAAAJ:u5HHmVD_uO8C), where a very large database of short speech fragments are recorded from a single speaker and then recombined to form complete utterances.  

Such an network architecture can be also applied to music generation, which is main focus of this book. It can generate unconditional music and conditional music. Of particular interest are conditional music models, which can generate music given a set of tags specifying e.g. genre or instruments. 

## Concepts

### Dilated Causal Convolution

The main ingredient of WaveNet are causal convolutions\(See Figure 1\). The word "causal" is used in the sense that the prediction $$p(x_{t+1} | x_1,...x_t)$$ emitted by the model at timestep $$t $$ cannot depend on any of the future time steps $$x_{t+1}, x_{t+2},...,x_{T}$$.

Because models with causal convolutions do not have recurrent connections, they are typically faster to train than RNNs, especially when applied to very long sequences.

There is a concept in the context of neural network called receptive field, specific to each node, is the spatial connectivity that this node can reach. Take the below image as example, the top right of output node can reach 5 elements in the input. For modeling the high frequency raw audio\(44.1 kHz is 44.1k samples per second\) we may want a large receptive field to cover long enough history.

![Figure 1: stack of causal convolutional layers](.gitbook/assets/figure1.png)

One of the problems of causal convolutions is that they require many layers, or large filters to increase the receptive field. To solve this problem WaveNet uses dilated convolution layers, this is a convolution where the filter is applied over an area larger than its length by skipping input values with a certain step. It is equivalent to a convolution with a larger filter derived from the original filter by dilating it with zeros, but is significantly more efficient.

![Figure 2: stack of dilated convolutional layers](.gitbook/assets/figure2.png)

In the paper it uses dilation of $$1,2,...,512,1,2...512,1,2...,512$$ layers. For example, each of 1,2,4,...,512 will have a receptive field of 1024 and can be an much more efficient than using $$1\times1024$$ convolution counterpart. Stacking these blocks will further enlarge the receptive field.

### Softmax Distributions and Companding Transformation

WaveNet uses softmax to model the conditional distributions $$p(x_{t+1} | x_1,...x_t)$$. The reason using a categorical distribution even for  continuous audio data is that a categorical distribution is more flexible and can more easily model arbitrary distributions because it makes no assumptions about their shape. 

Typically raw audio is stored in 16-bit integers for each time step which means our softmax layer will have 65,536 classes. To make this tractable, the author applied $$\mu$$ - law, which is a[ companding algorithm](https://en.wikipedia.org/wiki/%CE%9C-law_algorithm) to encode raw audio. This reduces the number of classes to predict down to 256. 

### Gated Activation Units

Normally people will add RELU as activation function to the output of a convolution layer.

$$z=RELU(W_{f,k}*x)$$ , where $$f$$ is denotes the filter, $$k$$ is layer index and $$ * $$ is convolution operator. 

In WaveNet, inspired by [PixelCNN](https://arxiv.org/pdf/1606.05328.pdf), the authors suggest using a Gated Activation instead of RELU as this yields significantly better result for modeling audio. This is denote by:

$$
z=tanh(W_{f,k})*x\odot\sigma(W_{g,k}*x)
$$

Here $$g$$ denotes gate and $$\odot$$ is the element-wise multiplication operator.

### Residual Learning and Skip Connections

The idea of using Residual Learning comes from [ResNet.](https://arxiv.org/abs/1512.03385) Residual learning is tackling the degradation problem which states that with the network depth increasing, accuracy gets saturated and then degrades rapidly. Surprisingly this is not due to overfitting as the training error also decreases. To illustrate this, consider we have a shallow network and deep network, the training error of deep network should never fall below its shallow counterparts as we can always just make the added layer identity to yield same result as the shallow one. That is to say the standard back-propagation has hard time to learn the identity mapping. 

So in ResNet the model is like this: suppose the desired mapping that we want is $$H(x) $$. Instead of optimizing against $$H(x)$$ we let our stacked non-linear layers to optimize over $$F(x):=H(x)-x$$. In the ResNet paper it hypothesized that it is easier to learn this residual mapping in comparison to identity mapping. 

In WaveNet, the 

