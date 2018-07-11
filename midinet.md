---
description: >-
  A Convolutional Generative Adversarial Network for Symbolic-domain Music
  Generation
---

# MidiNet

## Background and Introduction

### Related Work

By the time of MidiNet\(03/2017\) there are many deep-learning based music generation models, including WaveNet and MelodyRNN. But at that time majority of all endeavor was using RNN and their variants. WaveNet was the only major player who use CNN. One advantage of training CNN vs RNN is that the former is faster and more easily parallelizable. 

### The Use of GAN

Another major deep learning breakthrough during that period is GAN\(Generative Adversarial Network\). The use of GAN provides the needed creativity for music. MidiNet also use GAN architecture with CNN-like generator and discriminator. In the case of MidiNet the generator is to transform random noises into a 2-D scorelike representation, that “appears” to be from real MIDI. Meanwhile the discriminator takes this 2-D scorelike representation and predicts whether this is real or not.

### Conditioner CNN

They propose a novel conditional mechanism to use music from the previous bars to condition the generation of the present bar. They use another trainable CNN to incorporate information on the previous bars to some intermediate layers of Generator CNN. Without using recurrent unit as people used in RNN, we have some way to make use the past information.

The Conditioner, in general can encode any prior knowledge that people have on the music generation.

### Feature Matching

The idea of feature matching is proposed in the paper [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498v1.pdf). The idea is instead of optimize against the output layer of D, the generator G should optimize over the intermediate layers of D. That is to say we are minimizing the difference between some intermediate layers of G and D.

The use of feature matching is when we use Conditioner CNN, we can incorporate some manual tuning in the music generation such as we can make our music following a chord progression or following a priming melody.

## Method

### Symbolic Representation for Convolution

The input MIDI files is split by bars. For each bar, we use a $$h\times w$$ matrix where $$h$$ is the number of MIDI notes that we consider. In their implementation this value is set to be 128 which represents all the notes between C0 and G10. However in their training model, they shifts all the melodies to only 2 scales C4 to B5\(Although you still have all the representations for 128 notes\). They claim doing in this way they can more easily model collapse by detecting if any generated note is outside C4 and B5. Also they did not put another dimension for silence as their training data does not have any.

$$w$$ is the number of time steps in a bar. In the implementation it is set to be 16 which means the most granular note is a 16th note. If there is pause then it simply extends the last note.

In their Github they use\(at least part of\) the source code for DCGAN, so they treat audio file just like image files.

### Generator and Discriminator

For a normal GAN, the loss function is like the below one where you play a minimax game between Discriminator and Generator by alternatively optimize over them.

$$
\underset{G}{min}\underset{D}{max} V(D,G)=E_{\mathbf{X}\sim P_{data}(X)}[log(D(x))]+E_{\mathbf{Z}\sim P_{z}(z)}[log(1-D(G(z)))]
$$

Other than this typical setting,  the model also add a regularization term to the real and generated data\(include some intermediate level layers\) to enforce them to be close, the final loss also includes:

 $$\lambda_1||EX-EG(z)||_{2}^{2}+\lambda_2||Ef(X)-Ef(G(z))||_{2}^{2}$$ 

$$f$$ is the first convolution layer of discriminator and $$\lambda_1,\lambda_2$$ are hyper-parameters. By increasing them we can  make the music that we generate closer to existing music.

### Conditioner

GAN-like models often encode prior knowledge as some vector or matrix value that is to be added in some intermediate layer of Generator G and discriminator D.

The paper illustrate some cases of how to use tensor broadcasting to add tensors\(layers and prior knowledge\) of different shape. 

* For instance, if we have an additional information of shape n that we want to add to an intermediate layer of shape a-by-b. What we can do to the shape-n vector is we duplicate each value ab times to create a tensor of shape a-by-b-by-n, then concatenate this tensor to the intermediate layer, this is called 1-D conditions.
* If we use previous bar\(or bars\) as the condition, since our audio input has the shape of h-by-w\(see previous section for the Symbolic Representation for Convolution\). This is the case of 2-D conditions and we need a way to map a h-by-w matrix to a-by-b intermediate layer. So we use a trainable Conditioner CNN to do the trick. The conditioner and generator CNNs use exactly the same filter shapes in their convolution layers, so that the outputs of their convolution layers have “compatible” shapes.

## References

* Yang, Li-Chia, Szu-Yu Chou, and Yi-Hsuan Yang. "MidiNet: A convolutional generative adversarial network for symbolic-domain music generation." arXiv preprint arXiv:1703.10847\(2017\).
* Salimans, Tim, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. "Improved techniques for training gans." In Advances in Neural Information Processing Systems, pp. 2234-2242. 2016.

### 





