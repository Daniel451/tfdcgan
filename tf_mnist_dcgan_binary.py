import tensorflow as tf
import numpy as np
import argparse

import matplotlib as mpl
mpl.use("Agg")

from tensorflow.examples.tutorials.mnist import input_data

from helper import Helper as H


# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--mnist_data_path", type=str, default="", help="path to the MNIST data")
parser.add_argument("--img_save_path", type=str, default="", help="path where to store generated images")
parser.add_argument("--batch_size", type=int, default=100, help="integer, default is 100")
parser.add_argument("--epochs", type=int, default=100, help="integer, default is 100")
parser.add_argument("--mode", type=str, default="generate", choices=["train", "generate"])
args = parser.parse_args()

if args.img_save_path[-1] != "/":
    args.img_save_path += "/"




class NetClass(object):

    def __init__(self):
        self.model_save_path = args.img_save_path + "model.ckpt"

        self.Z = tf.placeholder(tf.float32, shape=[None, 100], name="Z")
        self.X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="X")
        self.Y = tf.placeholder(tf.int32, shape=[None], name="Y")

        self.G, self.G_logits = self.model_generator(self.Z, reuse=False)
        self.D, self.D_logits = self.model_discriminator(self.X, reuse=False)
        self.DG, self.DG_logits = self.model_discriminator(self.G, trainable=False)

        self.model_losses()
        self.initialize_network()


    def initialize_network(self):
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.init = tf.global_variables_initializer()
        self.session.run(self.init)


    def model_losses(self):
        # losses
        self.d_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.D_logits, labels=self.Y))
        self.dg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.DG_logits, labels=self.Y))

        self.d_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.999, name="Adam_D")\
            .minimize(self.d_loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="d"))
        self.g_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.999, name="Adam_DG")\
            .minimize(self.dg_loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="g"))


    def model_generator(self, Z, reuse=True):

        init_op = tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32)

        with tf.variable_scope("g", initializer=init_op, reuse=reuse, dtype=tf.float32):

            with tf.variable_scope("reshape"):
                out = tf.layers.dense(Z, 7 * 7 * 256, activation=None)
                out = tf.reshape(out, [-1, 7, 7, 256])
                out = tf.layers.batch_normalization(out)
                out = tf.nn.tanh(out)

            with tf.variable_scope("deconv1"):
                out = tf.layers.conv2d_transpose(out, 128, [3, 3], strides=[2, 2], padding="same")
                out = tf.layers.batch_normalization(out)
                out = tf.nn.tanh(out)

            with tf.variable_scope("deconv2"):
                out = tf.layers.conv2d_transpose(out, 64, [3, 3], strides=[2, 2], padding="same")
                out = tf.layers.batch_normalization(out)
                out = tf.nn.tanh(out)

            with tf.variable_scope("output"):
                out = tf.layers.conv2d_transpose(out, 1, [5, 5], strides=[1, 1], padding="same")
                logits = out
                output = tf.nn.tanh(out)

        return output, logits


    def model_discriminator(self, X, reuse=True, trainable=True):

        init_op = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)

        with tf.variable_scope("d", initializer=init_op, reuse=reuse, dtype=tf.float32):

            with tf.variable_scope("conv1"):
                out = tf.layers.conv2d(X, 64, [5, 5], strides=[2, 2], padding="same",
                                       trainable=trainable)
                out = tf.nn.tanh(out)

            with tf.variable_scope("conv2"):
                out = tf.layers.conv2d(out, 128, [3, 3], strides=[2, 2], padding="same",
                                       trainable=trainable)
                out = tf.nn.tanh(out)

            with tf.variable_scope("conv3"):
                out = tf.layers.conv2d(out, 256, [3, 3], strides=[1, 1], padding="same",
                                       trainable=trainable)
                out = tf.nn.tanh(out)

            with tf.variable_scope("output"):
                out = tf.reshape(out, [-1, 7 * 7 * 256])
                out = tf.layers.dense(out, 2, activation=None, trainable=trainable)
                logits = out
                output = tf.sigmoid(out)

        return output, logits


    def generate(self):
        print()
        print("loading network weights...")
        self.saver.restore(self.session, self.model_save_path)
        print("...done!")

        z = np.random.standard_normal((args.batch_size, 100)).astype(np.float32)
        gen_images = self.session.run(self.G, feed_dict={self.Z: z})

        plot_title = "generated image"
        fname = "img_generated_{}.jpg".format(H.get_random_str(5))
        print()
        print("writing image to {}...".format(args.img_save_path + fname))
        H.plot_batch_to_disk(gen_images, args.img_save_path, fname, plot_title)
        print("...done!")


    def train(self):
        # data
        mnist = input_data.read_data_sets(args.mnist_data_path, one_hot=True)

        batch_size = args.batch_size
        epochs = args.epochs
        train_dataset_size = 60000
        runs_per_epoch = int(train_dataset_size / batch_size)

        # label encoding fake [0, 1]
        # sparse labels
        labels_fake = np.ones(batch_size, dtype=np.int32)
        # labels_fake = np.zeros((batch_size, 2), dtype=np.float32)
        # labels_fake[:, 1] = 1.0

        # label encoding real [1, 0]
        # sparse labels
        labels_real = np.zeros(batch_size, dtype=np.int32)
        # labels_real = np.zeros((batch_size, 2), dtype=np.float32)
        # labels_real[:, 0] = 1.0

        for e in range(epochs):
            for run in range(runs_per_epoch):
                # MNIST training images
                train_images, test_images = mnist.train.next_batch(batch_size)
                # reshape and get images to an interval of [-1.0, 1.0]
                train_images = train_images.reshape((-1, 28, 28, 1)) * 2.0 - 1.0

                # train D on real images
                d_r_train, d_r_cost, debug_y_d_r, debug_D_r = self.session.run([self.d_train_op, self.d_loss, self.Y, self.D],
                                                                       feed_dict={self.X: train_images, self.Y: labels_real})

                # generate G(z) training data
                z = np.random.standard_normal((batch_size, 100)).astype(np.float32)
                gen_images = self.session.run(self.G, feed_dict={self.Z: z})

                # train D on fake images
                d_f_train, d_f_cost, debug_y_d_f, debug_D_f = self.session.run([self.d_train_op, self.d_loss, self.Y, self.D],
                                           feed_dict={self.X: gen_images, self.Y: labels_fake})

                # train Generator
                g_train, g_cost, debug_y_g, debug_G, debug_DG = self.session.run([self.g_train_op, self.dg_loss, self.Y, self.G,
                                                                         self.DG],
                        feed_dict={self.Z: z, self.Y: labels_real})


                if run % 50 == 0:
                    # evaluate
                    print()
                    H.print_costs(d_r_cost, "D real", e, run)
                    H.print_costs(d_f_cost, "D fake", e, run)
                    H.print_costs(g_cost, "G", e, run)

                    plot_title = "epoch {:0>3} run {:0>4}".format(e, run)
                    fname = "img_train_epoch_{:0>3}_run_{:0>4}.jpg".format(e, run)
                    H.plot_batch_to_disk(gen_images, args.img_save_path, fname, plot_title)


            # save graph once per epoch
            print()
            print("Saving model to {}...".format(self.model_save_path))
            self.saver.save(self.session, self.model_save_path)
            print("...done!")





if __name__ == "__main__":
    net = NetClass()
    if args.mode == "train":
        net.train()
    elif args.mode == "generate":
        net.generate()
    print("\nfinished!\n")
