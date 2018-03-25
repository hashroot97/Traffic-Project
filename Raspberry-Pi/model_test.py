import numpy as np
import tensorflow as tf
from numpy import random

training_data = np.load('training_data.npy')

x = tf.placeholder(tf.float32, shape=[None, 128, 128, 1])
y_true = tf.placeholder(tf.float32, shape=[None, 2])

y_true_cls = tf.argmax(y_true, axis=1)
conv_1 = tf.layers.conv2d(inputs=x, kernel_size=5, filters=16, padding='same', activation=tf.nn.relu)
max_pool_1 = tf.layers.max_pooling2d(inputs=conv_1, pool_size=2, strides=2)
conv_2 = tf.layers.conv2d(inputs=max_pool_1, kernel_size=5, filters=32, padding='same', activation=tf.nn.relu)
max_pool_2 = tf.layers.max_pooling2d(inputs=conv_2, pool_size=2, strides=2)
conv_3 = tf.layers.conv2d(inputs=max_pool_2, kernel_size=5, filters=64, padding='same', activation=tf.nn.relu)
max_pool_3 = tf.layers.max_pooling2d(inputs=conv_3, pool_size=2, strides=2)
conv_4 = tf.layers.conv2d(inputs=max_pool_3, kernel_size=5, filters=32, padding='same', activation=tf.nn.relu)
max_pool_4 = tf.layers.max_pooling2d(inputs=conv_4, pool_size=2, strides=2)
flat = tf.contrib.layers.flatten(inputs=max_pool_4)
dense_1 = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu)
dense_2 = tf.layers.dense(inputs=dense_1, units=128, activation=tf.nn.relu)
dense_3 = tf.layers.dense(inputs=dense_2, units=2, activation=None)
y_pred = tf.nn.softmax(logits=dense_3)
y_pred_cls = tf.argmax(y_pred, axis=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=dense_3)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
session = tf.Session()
saver = tf.train.Saver()
save_path = './checkpoints/model-best'
saver.restore(save_path=save_path, sess=session)
random_batch_x = []
random_batch_y = []
for i in range(100):
    rand_num = random.randint(0,2000)
    random_batch_x.append(training_data[rand_num][0])
    random_batch_y.append(training_data[rand_num][1])
random_batch_x = np.asarray(random_batch_x)
random_batch_y = np.asarray(random_batch_y)
print(random_batch_x.shape, random_batch_y.shape)

feed_dict = {x:random_batch_x, y_true:random_batch_y}
acc = session.run(accuracy, feed_dict=feed_dict)
print('accuracy : ', acc)
