import numpy as np
import tensorflow as tf

def store_4d_array(kernel, filename, scale):
    # store the kernel
    f = open(filename, 'w+')
    shape = kernel.shape
    num_out_channel = shape[3]
    num_in_channel = shape[2]
    num_width = shape[0]
    f.write(str(num_out_channel) + ',' + str(num_in_channel) + ',' + str(num_width) + ',' + str(num_width) + '\n')
    for index_out_channel in range(num_out_channel):
        for index_in_channel in range(num_in_channel):
            for index_row in range(num_width):
                for index_col in range(num_width):
                    f.write(str(kernel[index_row][index_col][index_in_channel][index_out_channel]*scale))
                    if index_col == num_width - 1:
                        f.write('\n')
                    else:
                        f.write(',')
    f.close()

def store_1d_2d_array(bias, filename):
    # store the bias
    bias = bias * 1000
    bias = bias.astype(int)
    np.savetxt(filename, bias, delimiter=',', fmt="%d")

if __name__ == "__main__":
    with tf.Session() as sess:
        # load the meta graph and weights
        saver = tf.train.import_meta_graph('model_2\minist.ckpt-70.meta')
        saver.restore(sess, tf.train.latest_checkpoint('model_2/'))
        # get weighs
        graph = tf.get_default_graph()
        conv1_w = sess.run(graph.get_tensor_by_name('conv1/w:0'))
        np.save("conv1_w", conv1_w)
        store_4d_array(conv1_w, "weights/conv1_w.txt")
        conv1_b = sess.run(graph.get_tensor_by_name('conv1/b:0'))
        store_1d_2d_array(conv1_b, "weights/conv1_b.txt")
        conv2_w = sess.run(graph.get_tensor_by_name('conv2/w:0'))
        store_4d_array(conv2_w, "weights/conv2_w.txt")
        conv2_b = sess.run(graph.get_tensor_by_name('conv2/b:0'))
        store_1d_2d_array(conv2_b, "weights/conv2_b.txt")
        fc1_w = sess.run(graph.get_tensor_by_name('fc1/w:0'))
        store_1d_2d_array(fc1_w, "weights/fc1_w.txt")
        fc1_b = sess.run(graph.get_tensor_by_name('fc1/b:0'))
        store_1d_2d_array(fc1_b, "weights/fc1_b.txt")
        fc2_w = sess.run(graph.get_tensor_by_name('fc2/w:0'))
        store_1d_2d_array(fc2_w, "weights/fc2_w.txt")
        fc2_b = sess.run(graph.get_tensor_by_name('fc2/b:0'))
        store_1d_2d_array(fc2_b, "weights/fc2_b.txt")