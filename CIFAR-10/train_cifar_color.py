# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
import gzip
import os
import sys
import tarfile
import tensorflow 

from tensorflow.models.image.cifar10 import cifar10_input
from tensorflow.python.framework import ops
from ops import BatchNorm, conv2d, deconv2d, linear, lrelu


#
# hyper parameters
#


tf.sg_verbosity(10)


batch_size = 32   # batch size
#batch_size = 32   # batch size

num_category = 10  # total categorical factor
num_cont = 4  # total continuous factor
num_dim = 110  # total latent dimension

def load_cifar10(data_dir,batch_size):
 
  data_dir = os.path.join(data_dir, 'cifar-10-batches-binary')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=batch_size)
  #images = tf.image.rgb_to_grayscale(images)
  return images, labels



# set log level to debug


data = load_cifar10('.',batch_size=32)
x = data[0]
y = tf.ones(batch_size, dtype=tf.sg_floatx)
train_labels = data[1]

# discriminator labels ( half 1s, half 0s )
y_disc = tf.concat(0, [y, y * 0])

#
# create generator
#

# random class number
z_cat = tf.multinomial(tf.ones((batch_size, num_category), dtype=tf.sg_floatx) / num_category, 1).sg_squeeze().sg_int()

# random seed = random categorical variable + random uniform
z = z_cat.sg_one_hot(depth=num_category).sg_concat(target=tf.random_uniform((batch_size, num_dim - num_category)))

# random continuous variable
z_cont = z[:, num_category:num_category+num_cont]

# category label
label = tf.concat(0, [train_labels, z_cat])


# generator network
with tf.sg_context(name='generator', size=5, stride=2, act='relu', bn=True):
    gen = (z.sg_dense(dim=1024)
           .sg_dense(dim=6*6*128)
           .sg_reshape(shape=(-1, 6, 6, 128))
           #.sg_upconv(dim=128)
           .sg_upconv(dim=64)           
           .sg_upconv(dim=3, act='tanh', bn=False)
           )


tf.sg_summary_image(gen)

# create real + fake image input
xx = tf.concat(0, [x, gen])
#
# create discriminator & recognizer
#



with tf.sg_context(name='discriminator', size=4, stride=2, act='leaky_relu'):
    # shared part
    shared = (xx.sg_conv(dim=16)
              .sg_conv(dim=32)
              .sg_conv(dim=64)
              .sg_conv(dim=128)
              .sg_conv(dim=256)
              .sg_flatten()
              .sg_dense(dim=1024))

    disc = shared.sg_dense(dim=1, act='sigmoid').sg_squeeze()

    # shared recognizer part
    recog_shared = shared.sg_dense(dim=128)

    # categorical auxiliary classifier end
    class_cat = recog_shared.sg_dense(dim=num_category, act='softmax')
    # continuous auxiliary classifier end
    class_cont = recog_shared[batch_size:, :].sg_dense(dim=num_cont, act='sigmoid')





#
# loss and train ops
#


loss_disc = tf.reduce_mean(disc.sg_bce(target=y_disc))  # discriminator loss
loss_gen = tf.reduce_mean(disc.sg_reuse(input=gen).sg_bce(target=y))  # generator loss
loss_class = tf.reduce_mean(class_cat.sg_ce(target=label)) \
             + tf.reduce_mean(class_cont.sg_mse(target=z_cont))  # recognizer loss

train_disc = tf.sg_optim(loss_disc + loss_class, lr=0.0001, category='discriminator')  # discriminator train ops
train_gen = tf.sg_optim(loss_gen + loss_class, lr=0.001, category='generator')  # generator train ops


#
# training
#

# def alternate training func
@tf.sg_train_func
def alt_train(sess, opt):
    l_disc = sess.run([loss_disc, train_disc])[0]  # training discriminator
    l_gen = sess.run([loss_gen, train_gen])[0]  # training generator
    return np.mean(l_disc) + np.mean(l_gen)

# do training
alt_train(log_interval=10, max_ep=500, ep_size=1562, early_stop=False)
