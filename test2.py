import numpy as np
import tensorflow as tf
import os
from os.path import join as pjoin
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print(tf.__version__)



x_train = np.load("./info/task1/x_train.npy").reshape([1500, 7500])
y_train = np.load("./info/task1/y_train.npy")
y_train = tf.keras.utils.to_categorical(y_train)

# 445
x_test = np.load("./info/task1/x_test.npy").reshape([289, 7500])
y_test = np.load("./info/task1/y_test.npy")
y_test = tf.keras.utils.to_categorical(y_test)


x_train = x_train * 1.0 / 127.5 - 1
x_test = x_test * 1.0 / 127.5 - 1
print("data load finish")

# Parameters
learning_rate = 0.01
training_epochs = 500
batch_size = 1500
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 7500]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 3]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([7500, 3]))
b = tf.Variable(tf.zeros([3]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(x_train.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = x_train
            batch_ys = y_train
            # Fit training using batch data
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy for 3000 examples
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: x_test, y: y_test}))