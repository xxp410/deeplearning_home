from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import os

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
# Parameters
learning_rate = 0.001
training_epochs = 500
batch_size = 100
display_step = 1
is_train = 1
# Network Parameters
n_input = 784
n_classes = 10
# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
# pre-define
def conv2d(x, W):
    return tf.nn.conv2d(x, W,strides=[1, 1, 1, 1],padding='SAME', name='conv_2d')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME', name='max_pool')

# Create model
def multilayer_preceptron(x, weights, biases):
    # now,we want to change this to a CNN network
    # first,reshape the data to 4_D ,
    x_image = tf.reshape(x, [-1, 28, 28, 1],name='reshape')
    # then apply cnn layers ,cnn layer and activation function --relu
    h_conv1 = tf.nn.relu(conv2d(x_image, weights['conv1']) + biases['conv_b1'], name='relu')
    # first pool layer
    h_pool1 = max_pool_2x2(h_conv1)
    # second cnn layer
    h_conv2 = tf.nn.relu(conv2d(h_pool1, weights['conv2']) + biases['conv_b2'], name='relu')
    # second pool layer
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64],name='reshape')
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights['fc1'],name='matmul') + biases['fc1_b'],name='relu')
    out_layer = tf.matmul(h_fc1, weights['out'], name='matmul') + biases['out_b']
    return out_layer

weights = {
    'conv1': tf.Variable(tf.random_normal([5, 5, 1, 32]),name='conv1_w'),
    'conv2': tf.Variable(tf.random_normal([5, 5, 32, 64]),name='conv2_w'),
    'fc1': tf.Variable(tf.random_normal([7 * 7 * 64, 256]),name='fc1_w'),
    'out': tf.Variable(tf.random_normal([256, n_classes]),name='out_w')
}

biases = {
    'conv_b1': tf.Variable(tf.random_normal([32]),name='conv_b1'),
    'conv_b2': tf.Variable(tf.random_normal([64]),name='conv_b2'),
    'fc1_b': tf.Variable(tf.random_normal([256]),name='fc1_b'),
    'out_b': tf.Variable(tf.random_normal([n_classes]),name='out_b')
}

# Construct model
pred = multilayer_preceptron(x, weights, biases)
if is_train:
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y,name='cross_entropy'),name='reduce_mean')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    trainer = optimizer.minimize(cost)
    # Initializing the variables
    init = tf.global_variables_initializer()
# saver model
model_saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    # 训练阶段
    if is_train:
        sess.run(init)
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # run optimization op (backprop)and cost op (to get loss value)
                _, c = sess.run([trainer, cost], feed_dict={x: batch_x, y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
                # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                # Calculate accuracy
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                accuracy_val = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
                print("Accuracy:", accuracy_val)
                if accuracy_val > 0.93:
                    break;
        print("Optimization Finished!")
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

        writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
        writer.close()
        variable_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variable_names)
        for k, v in zip(variable_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)
            print(v)
        # create dir for model saver
        model_dir = "mnist"
        model_name = "cpk"

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_saver.save(sess, os.path.join(model_dir, model_name))
        print("model saved sucessfully")
    else:
        # create dir for model saver
        model_dir  = "mnist"
        model_name = "cpk"
        test_index = 100
        for i in range(test_index):
            model_path = os.path.join(model_dir, model_name)
            model_saver.restore(sess, model_path)
            img = mnist.test.images[i].reshape(-1, 784)
            img_show = mnist.test.images[i].reshape(28, 28)
            img_label = sess.run(tf.argmax(mnist.test.labels[i]))
            ret = sess.run(pred, feed_dict={x: img})
            num_pred = sess.run(tf.argmax(ret, 1))
            plt.imshow(img_show, cmap='gray')
            plt.xlabel("predict value:%d,   real value:%d\n" %(num_pred,img_label))
            print("预测值:%d\n" %num_pred)
            print("真实值:", img_label)
            print("模型恢复成功")
            plt.pause(2)
