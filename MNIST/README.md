# ACGAN on MNIST
forked from https://github.com/buriburisuri/ac-gan

## Training the network

Execute
<pre><code>
python train.py
</code></pre>
to train the network. You can see the result ckpt files and log files in the 'asset/train' directory.
Launch tensorboard --logdir asset/train/log to monitor training process.


## Generating image
 
Execute
<pre><code>
python generate.py
</code></pre>
to generate sample image.  The 'sample.png' file will be generated in the 'asset/train' directory.