# ACGAN on CIFAR-10


Dataset from https://www.cs.toronto.edu/~kriz/cifar.html

## Training the network

Execute
<pre><code>
python train_cifar_color.py
</code></pre>
to train the network. You can see the result ckpt files and log files in the 'asset/train' directory.
Launch tensorboard --logdir asset/train/log to monitor training process.

##  Generating image
 
Execute
<pre><code>
python generate_cifar_color.py
</code></pre>
to generate sample image.  Images files will be generated in the 'asset/train' directory.
