# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
import os
import tensorflow 
import load_chars74k


from scipy import ndimage, misc
import pandas as pd



# set log level to debug
tf.sg_verbosity(10)
#
# hyper parameters
#

batch_size = 32   # batch size
num_category = 62 # total categorical factor
num_cont = 2  # total continuous factor
num_dim = 150  # total latent dimension
px = 32

#
# Input Data
#
### MAIN ######
# load mixed english handwritten data###

data = load_chars74k.Chars74k(batch_size=batch_size)

# input images
x = data.train.image

y = tf.ones(batch_size, dtype=tf.sg_floatx)


# discriminator labels ( half 1s, half 0s )
y_disc = tf.concat(0, [y, y * 0])

#
# create generator
#

# random class number
z_cat = tf.multinomial(tf.ones((batch_size, num_category), dtype=tf.sg_floatx)/num_category, 1).sg_squeeze().sg_int()

# random seed = random categorical variable + random uniform
z = z_cat.sg_one_hot(depth=num_category).sg_concat(target=tf.random_uniform((batch_size, num_dim - num_category)))

# random continuous variable
z_cont = z[:, num_category:num_category+num_cont]

label = tf.concat(0, [data.train.label, z_cat]) 


# generator network
with tf.sg_context(name='generator', size=5, stride=2, act='relu', bn=True):
    gen = (z.sg_dense(dim=1024)
            .sg_dense(dim=8*8*128)
            .sg_reshape(shape=(-1, 8, 8, 128))
            .sg_upconv(dim=64)
            .sg_upconv(dim=1, act='sigmoid', bn=False))



print ('gen_shape:') 
print (gen.get_shape())

# add image summary
tf.sg_summary_image(gen)

#
# create discriminator & recognizer
#

# create real + fake image input
x = x.sg_reshape(shape=(-1,32,32,1))
xx = tf.concat(0, [x, gen])

with tf.sg_context(name='discriminator', size=4, stride=2, act='leaky_relu',bn=True):
    # shared part
    shared = (xx.sg_conv(dim=32)
                .sg_conv(dim=64)
                .sg_conv(dim=128)
                .sg_flatten()
                .sg_dense(dim=1024))

    # discriminator end
    disc = shared.sg_dense(dim=1, act='linear').sg_squeeze()

    # shared recognizer part
    recog_shared = shared.sg_dense(dim=128)

    # categorical auxiliary classifier end
    class_cat = recog_shared.sg_dense(dim=num_category, act='linear')
    # continuous auxiliary classifier end
    class_cont = recog_shared[batch_size:, :].sg_dense(dim=num_cont, act='sigmoid')

#
# loss and train ops
#

loss_disc = tf.reduce_mean(disc.sg_bce(target=y_disc))  # discriminator loss
loss_gen = tf.reduce_mean(disc.sg_reuse(input=gen).sg_bce(target=y))  # generator loss
loss_class = tf.reduce_mean(class_cat.sg_ce(target=label)) \
             + tf.reduce_mean(class_cont.sg_mse(target=z_cont))  # recognizer loss

train_disc = tf.sg_optim(loss_disc + loss_class, lr=0.00005, category='discriminator')  # discriminator train ops
train_gen = tf.sg_optim(loss_gen + loss_class, lr=0.0005, category='generator')  # generator train ops


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
alt_train(log_interval=10, max_ep=1500, ep_size=data.train.num_batch, early_stop=False)
