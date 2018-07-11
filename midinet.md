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

## Code Review

The official code provided by the authors can be found [here in Github](https://github.com/RichardYang40148/MidiNet/tree/master/v1). The implementation is written in TensorFlow. The code used the structure of the [DCGAN code base](https://github.com/carpedm20/DCGAN-tensorflow) with modified Discriminator and Generator which takes conditioning into consideration

### Discriminator

```python
# how the discrinimator get called in GAN
self.D, self.D_logits, self.fm = self.discriminator(self.images, self.y, reuse=False)
```

```python
# This part is how discriminator get defined. 
# y here is the additional information(prior knowledge) that we want to add.
# y_dim is the dimension of y.
def discriminator(self, x, y=None, reuse=False):
    df_dim = 64
    dfc_dim = 1024
    if reuse:
        tf.get_variable_scope().reuse_variables()

    yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
    # this is doing the 1-D conditoning
    # we concat yb with input x
    x = conv_cond_concat(x, yb)
    h0 = lrelu(conv2d(x, self.c_dim + self.y_dim,k_h=2, k_w=128, name='d_h0_conv'))
    fm = h0
    # conditioning is also implemented on 1st layer
 
       h0 = conv_cond_concat(h0, yb)

    h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim,k_h=4, k_w=1, name='d_h1_conv')))
    h1 = tf.reshape(h1, [self.batch_size, -1])            
    h1 = tf.concat(1, [h1, y])
    
    h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
    h2 = tf.concat(1, [h2, y])

    h3 = linear(h2, 1, 'd_h3_lin')
    
    return tf.nn.sigmoid(h3), h3, fm
```

### Generator

```python
# how Generator get called in GAN
self.G = self.generator(self.z, self.y, self.prev_bar)
```

```python
# z is the random vector to generate 
# y is the additional information
# prev_x is the same shape with input in the discriminator
def generator(self, z, y=None, prev_x = None):
    # Let's assume we have a mirrored encoder-decoder CNN model
    # The encoder takes prev_x, and create all intermediate layers 
    # Later on, decoder will start with random vector z,
    # For all the intermediate layers from decoder, we have a mirrored layer in the encoder, say h1_prev
    # This makes it possible to concat them. If we concat them on 3rd dimension, it looks like we just add some additional feature maps
    h0_prev = lrelu(self.g_prev_bn0(conv2d(prev_x, 16, k_h=1, k_w=128,d_h=1, d_w=2, name='g_h0_prev_conv')))
    h1_prev = lrelu(self.g_prev_bn1(conv2d(h0_prev, 16, k_h=2, k_w=1, name='g_h1_prev_conv')))
    h2_prev = lrelu(self.g_prev_bn2(conv2d(h1_prev, 16, k_h=2, k_w=1, name='g_h2_prev_conv')))
    h3_prev = lrelu(self.g_prev_bn3(conv2d(h2_prev, 16, k_h=2, k_w=1, name='g_h3_prev_conv')))

    # prior knowledege is added to random vector z
    yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
    z = tf.concat(1, [z, y])

    h0 = tf.nn.relu(self.g_bn0(linear(z, 1024, 'g_h0_lin')))
    h0 = tf.concat(1, [h0, y])

    h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*2*1, 'g_h1_lin')))
    h1 = tf.reshape(h1, [self.batch_size, 2, 1, self.gf_dim * 2])
    # here we conditioning on both prior knowledge and prev bar
    h1 = conv_cond_concat(h1, yb)
    h1 = conv_prev_concat(h1, h3_prev)
    # conditoning and prev bar information is add to every layer
    h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, 4, 1, self.gf_dim * 2],k_h=2, k_w=1,d_h=2, d_w=2 ,name='g_h2')))
    h2 = conv_cond_concat(h2, yb)
    h2 = conv_prev_concat(h2, h2_prev)

    h3 = tf.nn.relu(self.g_bn3(deconv2d(h2, [self.batch_size, 8, 1, self.gf_dim * 2],k_h=2, k_w=1,d_h=2, d_w=2 ,name='g_h3')))
    h3 = conv_cond_concat(h3, yb)
    h3 = conv_prev_concat(h3, h1_prev)

    h4 = tf.nn.relu(self.g_bn4(deconv2d(h3, [self.batch_size, 16, 1, self.gf_dim * 2],k_h=2, k_w=1,d_h=2, d_w=2 ,name='g_h4')))
    h4 = conv_cond_concat(h4, yb)
    h4 = conv_prev_concat(h4, h0_prev)

    return tf.nn.sigmoid(deconv2d(h4, [self.batch_size, 16, 128, self.c_dim],k_h=1, k_w=128,d_h=1, d_w=2, name='g_h5'))

```

The trick here is to incorporate information on the previous bar to every layer of the Generator network. We first do a feedforward start from prev\_x to create tensors h0\_prev through h3\_prev. Then we start from a random vector z. First of all we incorporate some prior knowledge y or equivalently y\_b. We do a deconv on the concat\(z, y\), this will create an intermediate layer of CNN which has the same shape\(at least for the first two dimensions\) with h0\_prev, which is also one intermediate layer of the opposite version of the CNN. This makes it possible to concatenate these two tensors on the channel dimension. We repeat this steps several times for all layers. This has the effect of what I described in the Conditioner section above. 



## References

* Yang, Li-Chia, Szu-Yu Chou, and Yi-Hsuan Yang. "MidiNet: A convolutional generative adversarial network for symbolic-domain music generation." arXiv preprint arXiv:1703.10847\(2017\).
* Salimans, Tim, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. "Improved techniques for training gans." In Advances in Neural Information Processing Systems, pp. 2234-2242. 2016.

### 





