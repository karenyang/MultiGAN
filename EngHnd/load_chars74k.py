from __future__ import absolute_import
import sugartensor as tf
import numpy as np
from scipy import ndimage, misc
import pandas as pd
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


__author__ = 'mansour'

class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=True,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=0):

  num_classes = 62

  dataset = pd.read_csv('training_data_mixed.csv', header=None).values
  #print 'dataset: ', dataset.shape
  images = dataset[:, 1:].astype(np.float32)
  train_images = np.multiply(images, 1.0 / 255.0)
  labels = dataset[:, 0]

  train_labels = labels

  #train_images = images.reshape([-1,32,32,1])

  print (train_images.shape)



  if not 0 <= validation_size <= len(train_images):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))

  #validation_images = train_images[:validation_size]
  #validation_labels = train_labels[:validation_size]
  #train_images = train_images[validation_size:]
  #train_labels = train_labels[validation_size:]

  train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
  validation = None #DataSet(validation_images,
                    #   validation_labels,
                    #   dtype=dtype,
                    #   reshape=reshape)
  #test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)
  test = None
  return base.Datasets(train=train, validation=validation, test=test)





# constant sg_data to tensor conversion with queue support
def _data_to_tensor(data_list, batch_size, name=None):
    r"""Returns batch queues from the whole data. 
    
    Args:
      data_list: A list of ndarrays. Every ndarray must have the same size in the first dimension.
      batch_size: An integer.
      name: A name for the operations (optional).
      
    Returns:
      A list of tensors of `batch_size`.
    """
    # convert to constant tensor
    const_list = [tf.constant(data) for data in data_list]

    # create queue from constant tensor
    queue_list = tf.train.slice_input_producer(const_list, capacity=batch_size*128, name=name)

    # create batch queue
    return tf.train.shuffle_batch(queue_list, batch_size, capacity=batch_size*128,
                                  min_after_dequeue=batch_size*32, name=name)


class Chars74k(object):
  r"""Downloads Mnist datasets and puts them in queues.
  """
  

  def __init__(self, batch_size=32, reshape=False, one_hot=True):
      _data_dir = '.'
      # load sg_data set
      data_set = read_data_sets(_data_dir, reshape=reshape, one_hot=one_hot)

      self.batch_size = batch_size

      # save each sg_data set
      _train = data_set.train

      #_valid = data_set.validation
      #_test = data_set.test

      # member initialize
      self.train, self.valid, self.test = tf.sg_opt(), tf.sg_opt, tf.sg_opt()

      # convert to tensor queue
      self.train.image, self.train.label = \
          _data_to_tensor([_train.images, _train.labels.astype('int32')], batch_size, name='train')
      #self.valid.image, self.valid.label = \
      #    _data_to_tensor([_valid.images, _valid.labels.astype('int32')], batch_size, name='valid')
      #self.test.image, self.test.label = \
      #    _data_to_tensor([_test.images, _test.labels.astype('int32')], batch_size, name='test')

      # calc total batch count
      self.train.num_batch = _train.labels.shape[0] // batch_size
      #self.valid.num_batch = _valid.labels.shape[0] // batch_size
      #self.test.num_batch = _test.labels.shape[0] // batch_size



