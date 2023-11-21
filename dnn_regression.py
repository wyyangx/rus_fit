
import numpy as np
import tensorflow as tf
import os
import time
import logging
from scipy import io


def bn(x, is_training, name):
    return batch_norm(x, is_training, name, update_mean_var=True, decay=0.99, epsilon=1e-3)
    # return tf.contrib.layers.batch_norm(x,
    #                                     decay=0.99,
    #                                     updates_collections=tf.GraphKeys.UPDATE_OPS,
    #                                     epsilon=0.001,
    #                                     scale=True,
    #                                     fused=False,
    #                                     is_training=is_training,
    #                                     scope=name)


def batch_norm(x, is_training, name, update_mean_var=True, decay=0.99, epsilon=1e-3):
    """
    :param x: input with shape=[batch_size，Height，Width，Channels]
    :param is_training: is it in training? Value = True or False
    :param name:
    :param update_mean_var: 是否更新均值方差这两个参数
    :param decay:
    :param epsilon:
    :return:
    """
    phase_update = tf.convert_to_tensor(update_mean_var, dtype=tf.bool)
    phase_train = tf.convert_to_tensor(is_training, dtype=tf.bool)
    axis = list(range(len(x.get_shape()) - 1))
    with tf.variable_scope(name):
        n_out = int(x.get_shape()[len(x.get_shape()) - 1])

        beta = tf.get_variable(name='beta', initializer=tf.constant(0.0, shape=[n_out], dtype=x.dtype),
                               trainable=True, dtype=x.dtype)
        gamma = tf.get_variable(name='gamma', initializer=tf.constant(1.0, shape=[n_out], dtype=x.dtype),
                                trainable=True, dtype=x.dtype)

        running_mean = tf.get_variable(name='running_mean',
                                       initializer=tf.constant(0.0, shape=[n_out], dtype=x.dtype),
                                       trainable=False, dtype=x.dtype)
        running_var = tf.get_variable(name='running_var',
                                      initializer=tf.constant(1.0, shape=[n_out], dtype=x.dtype),
                                      trainable=False, dtype=x.dtype)

        x_mean, x_var = tf.nn.moments(x, axis)

        update_mean = tf.cond(tf.logical_and(phase_update, phase_train),  # 训练 且 更新参数为True，才更新均值方差
                lambda: tf.assign(running_mean,
                                  tf.add(tf.multiply(running_mean, decay), tf.multiply(tf.identity(x_mean), 1 - decay))),
                lambda: tf.assign(running_mean, running_mean))

        update_var = tf.cond(tf.logical_and(phase_update, phase_train),
                lambda: tf.assign(running_var,
                                  tf.add(tf.multiply(running_var, decay), tf.multiply(tf.identity(x_var), 1 - decay))),
                lambda: tf.assign(running_var, running_var))

        with tf.control_dependencies([update_mean, update_var]):
            mean, var = tf.cond(phase_train, lambda: (tf.identity(x_mean), tf.identity(x_var)),
                                lambda: (tf.identity(running_mean), tf.identity(running_var)))

            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)

    return normed


def active_fun(x, leak=0.2, name='active_fun', type='sigmoid'):
    if type == 'lrelu':
        return tf.maximum(x, leak*x, name=name)
    elif type == 'tanh':
        return tf.nn.tanh(x, name=name)
    elif type == 'relu':
        return tf.maximum(x, 0, name=name)
    else:
        return tf.sigmoid(x, name=name)


def relu(x, name='active_fun'):
    # return tf.maximum(x, 0, name=name)
    return tf.maximum(x, 0.2 * x, name=name)


def dense(x, output_size, stddev=0.5, bias_start=0.0, reuse=False, name='dense', alpha=0.0001):
    shape = x.get_shape().as_list()
    with tf.variable_scope(name, reuse=reuse):
        W = tf.get_variable(
            'weights', [shape[1], output_size],
            tf.float32,
            tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable(
            'biases', [output_size],
            initializer=tf.constant_initializer(bias_start))

        # if alpha > 0:
        #     tf.add_to_collection('losses', tf.contrib.layers.l1_regularizer(alpha)(W))
        #     tf.add_to_collection('losses', tf.contrib.layers.l1_regularizer(alpha)(bias))

        out = tf.matmul(x, W) + bias
    return out


def diagonal(x, weight_start=1.0, bias_start=0.0, reuse=False, name='diagonal'):
    shape = x.get_shape().as_list()
    with tf.variable_scope(name, reuse=reuse):
        W = tf.get_variable('w', [shape[1]], tf.float32,
                            initializer=tf.constant_initializer(weight_start))

        B = tf.get_variable('b', [shape[1]], tf.float32,
                            initializer=tf.constant_initializer(bias_start))

        out = tf.matmul(x, tf.diag(W))
    return out


def res_block(x, output_size, name, af_type, stddev=1, bias_start=0.0, is_training=True, reuse=False):
    out = active_fun(bn(dense(x, output_size, stddev=stddev, bias_start=bias_start, reuse=reuse, name=name+"_w1"),
                        is_training,
                        name=name+"bn1"),
                     name=name+"a1", type=af_type)
    out = dense(out, output_size, stddev=stddev, bias_start=bias_start, reuse=reuse, name=name + "_w2") + x
    out = bn(out, is_training, name=name + "bn2")
    out = active_fun(out, name=name+"a2", type=af_type)
    return out


# dropout函数实现
def dropout(x, level):  # level为神经元保留的概率值，在0-1之间
    if level < 0. or level >= 1:
        raise Exception('Dropout level must be in interval [0, 1[.')
    retain_prob = level
    # 利用binomial函数，生成与x一样的维数向量。
    # 神经元x保留的概率为p，n表示每个神经元参与随机实验的次数，通常为1,。
    # size是神经元总数。
    sample = np.random.binomial(n=1, p=retain_prob, size=x.shape)
    # 生成一个0、1分布的向量，0表示该神经元被丢弃
    x *= sample
    # x /= retain_prob
    return x


LNodes = []
af_type = 'sigmoid'


# Building the network
def fit_net(input, output_dim, is_training, reuse, name="fit_net", alpha=0.0001):
    with tf.variable_scope(name, reuse=reuse):
        # the conventional net
        # l = 0
        # net = input
        # for lu in LNodes:
        #     l += 1
        #     net = active_fun(bn(dense(net, lu, name="fc%d" % l, alpha=alpha),
        #                         is_training, name="bn%d" % l), name="a%d" % l, type=af_type)
        #
        # l += 1
        # net = dense(net, output_dim, name="fc%d_1" % l, alpha=alpha)

        # the proposed net
        l = 0
        net = input
        for lu in LNodes:
            l += 1
            net = active_fun(bn(dense(net, lu, name="fc%d" % l, alpha=alpha),
                                is_training, name="bn%d" % l), name="a%d" % l, type=af_type)
            net = tf.concat([net, input], 1)

        l += 1
        net = dense(net, output_dim, name="fc%d_1" % l, alpha=alpha)
    return net


def shuffle_lists(xs, ys, num):
    shape = xs.shape
    ri = np.random.permutation(shape[0])
    ri = ri[0: num]
    batch_xs = np.empty((0, xs.shape[1]))
    batch_ys = np.empty((0, ys.shape[1]))
    for i in ri:
        batch_xs = np.vstack((batch_xs, xs[i]))
        batch_ys = np.vstack((batch_ys, ys[i]))

    return batch_xs, batch_ys


def shuffle_data(xs, ys):
    shape = xs.shape
    ri = np.random.permutation(shape[0])
    batch_xs = xs[ri, :]
    batch_ys = ys[ri, :]

    return batch_xs, batch_ys


def split_data(xs, ys, fold, k, disruption=True):
    shape = xs.shape
    if disruption:
        ri = np.random.permutation(shape[0])
        xs = xs[ri, :]
        ys = ys[ri, :]

    beg_i = xs.shape[0] // fold * k
    end_i = xs.shape[0] // fold * (k + 1)

    index = []
    for i in range(xs.shape[0]):
        index.append(i)

    index[beg_i:end_i] = []

    xs0 = xs[index, :]
    ys0 = ys[index, :]
    xs1 = xs[beg_i:end_i, :]
    ys1 = ys[beg_i:end_i, :]

    return xs0, ys0, xs1, ys1


class DnnRegression:
    def __init__(self, model='', alpha=0.0, batch_size=128):
        self.input_d = -1
        self.output_d = -1
        self.freq_index = []
        self.alpha = alpha
        self.batch_size = batch_size

        self.X = np.empty([0, 1])
        self.Y = np.empty([0, 1])
        self.X_train = np.empty([0, 1])
        self.Y_train = np.empty([0, 1])
        self.X_valid = np.empty([0, 1])
        self.Y_valid = np.empty([0, 1])
        self.X_test = np.empty([0, 1])
        self.Y_test = np.empty([0, 1])

        self.X_mean = np.empty([0, 1])
        self.X_var = np.empty([0, 1])
        self.Y_mean = np.empty([0, 1])
        self.Y_var = np.empty([0, 1])

        if not os.path.exists("./params"):
            os.mkdir("./params")

        if model == '':
            self.model_name = "params/dnn_regression"
        else:
            self.model_name = "params/dnn_regression_%s" % model

    def load_data(self, X, Y, fold, k):
        print("Precondition data ... ...")
        self.X = X
        self.Y = Y

        xs, ys, self.X_test, self.Y_test = split_data(X, Y, fold, k, False)
        self.X_train, self.Y_train, self.X_valid, self.Y_valid = split_data(xs, ys, 5, 1, True)

        self.input_d = X.shape[1]
        self.output_d = Y.shape[1]

        # 求均值方差
        self.X_mean = np.mean(self.X_train, axis=0)
        self.X_var = np.sqrt(np.var(self.X_train, axis=0))

        self.Y_mean = np.mean(self.Y_train, axis=0)
        self.Y_var = np.sqrt(np.var(self.Y_train, axis=0))

        np.save(self.model_name + "_Y_mean.npy", self.Y_mean)  # 保存为.npy格式
        np.save(self.model_name + "_Y_var.npy", self.Y_var)  # 保存为.npy格式

        np.save(self.model_name + "_X_mean.npy", self.X_mean)  # 保存为.npy格式
        np.save(self.model_name + "_X_var.npy", self.X_var)  # 保存为.npy格式

        self.X_train = self.X_train - np.tile(self.X_mean, (self.X_train.shape[0], 1))
        self.X_train = self.X_train / np.tile(self.X_var, (self.X_train.shape[0], 1))
        self.Y_train = self.Y_train - np.tile(self.Y_mean, (self.Y_train.shape[0], 1))
        self.Y_train = self.Y_train / np.tile(self.Y_var, (self.Y_train.shape[0], 1))

        self.X_valid = self.X_valid - np.tile(self.X_mean, (self.X_valid.shape[0], 1))
        self.X_valid = self.X_valid / np.tile(self.X_var, (self.X_valid.shape[0], 1))
        self.Y_valid = self.Y_valid - np.tile(self.Y_mean, (self.Y_valid.shape[0], 1))
        self.Y_valid = self.Y_valid / np.tile(self.Y_var, (self.Y_valid.shape[0], 1))

        self.X_test = self.X_test - np.tile(self.X_mean, (self.X_test.shape[0], 1))
        self.X_test = self.X_test / np.tile(self.X_var, (self.X_test.shape[0], 1))
        self.Y_test = self.Y_test - np.tile(self.Y_mean, (self.Y_test.shape[0], 1))
        self.Y_test = self.Y_test / np.tile(self.Y_var, (self.Y_test.shape[0], 1))

    def deep_net(self):
        tf.reset_default_graph()

        # 输入
        input_x = tf.placeholder(tf.float32, [None, self.input_d])
        out_y = tf.placeholder(tf.float32, [None, self.output_d])

        print(LNodes)
        # Construct model
        fit_op_train = fit_net(input_x, self.output_d, is_training=True, reuse=False, name='fit_net')
        fit_op_test = fit_net(input_x, self.output_d, is_training=False, reuse=True, name='fit_net')

        train_loss = tf.reduce_mean(tf.pow(out_y - fit_op_train, 2))
        eval_loss = tf.reduce_mean(tf.pow(out_y - fit_op_test, 2))

        if self.alpha > 0:
            var_lists = tf.trainable_variables()
            reg_ws = tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(self.alpha), var_lists)
            train_loss = train_loss + reg_ws
            eval_loss = eval_loss + reg_ws

        learning_rate = 0.0001  #

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_loss)

        return fit_op_train, fit_op_test, input_x, out_y, eval_loss, optimizer

    def train_net(self, epochs, skip_n=1000):
        print("Train net ... ...")
        fit_op_train, fit_op_test, input_x, out_y, eval_loss, optimizer = self.deep_net()
        saver = tf.train.Saver()

        batch_size = self.batch_size

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            min_loss = 1E5
            min_loss_epoch_n = 0
            for e in range(epochs):
                print('====>>>Epoch %i' % e)
                xs, ys = shuffle_data(self.X_train, self.Y_train)
                for i in range(self.X_train.shape[0] // batch_size):
                    beg_i = batch_size * i
                    end_i = batch_size * (i + 1)
                    bxs = xs[beg_i:end_i, :]
                    bys = ys[beg_i:end_i, :]

                    # Run optimization op (backprop) and cost op (to get loss value)
                    sess.run([optimizer], feed_dict={input_x: bxs, out_y: bys})

                if (e + 1) < skip_n:
                    continue

                if (e + 1) % 5 != 0:
                    continue

                # Validating
                loss_l = []
                for i in range(self.Y_valid.shape[0] // batch_size):
                    beg_i = batch_size * i
                    end_i = batch_size * (i + 1)
                    bxs = self.X_valid[beg_i:end_i, :]
                    bys = self.Y_valid[beg_i:end_i, :]

                    l = sess.run(eval_loss, feed_dict={input_x: bxs, out_y: bys})
                    loss_l.append(l)

                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<Epoch %i>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>' % e)
                np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
                mean_loss = np.mean(np.array(loss_l))
                print('mean lost: ', mean_loss)
                print('min_loss: ', min_loss)

                if mean_loss < 0.0008:  #
                    break

                if min_loss > mean_loss:
                    min_loss = mean_loss
                    saver.save(sess, self.model_name + ".ckpt")
                    min_loss_epoch_n = 0
                else:
                    min_loss_epoch_n += 1

                if min_loss_epoch_n >= 20:
                    print("Loss = %f" % min_loss)
                    break

            return min_loss

    def test_net(self):
        print("Test net ... ...")
        fit_op_train, fit_op_test, input_x, out_y, eval_loss, optimizer = self.deep_net()
        saver = tf.train.Saver()

        batch_size = self.batch_size

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.model_name + ".ckpt")  # 加载训练好的网络参数

            res_counts = np.zeros([6, np.sum(self.output_d)])
            relative_error = np.empty([0, self.output_d])
            error = np.empty([0, self.output_d])

            for i in range(self.Y_test.shape[0] // batch_size):
                beg_i = batch_size * i
                end_i = batch_size * (i + 1)
                bxs = self.X_test[beg_i:end_i, :]
                bys = self.Y_test[beg_i:end_i, :]
                #
                out_ys = sess.run(fit_op_test, feed_dict={input_x: bxs})
                #
                out_ys = out_ys * self.Y_var + self.Y_mean
                real_ys = bys * self.Y_var + self.Y_mean

                e = real_ys - out_ys
                error = np.vstack((error, e))

                re = 100 * np.abs(e) / real_ys
                relative_error = np.vstack((relative_error, re))

                np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
                for n in range(re.shape[0]):
                    for p in range(re.shape[1]):
                        if re[n, p] < 0.05:
                            index = 0
                        elif re[n, p] < 0.1:
                            index = 1
                        elif re[n, p] < 0.2:
                            index = 2
                        elif re[n, p] < 0.5:
                            index = 3
                        elif re[n, p] < 1.0:
                            index = 4
                        else:
                            index = 5
                        res_counts[index, p] += 1

            count = np.sum(res_counts, axis=0)
            count = count[0]
            res_counts = 100 * (res_counts / count)
            np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
            print(res_counts)
            print('relative_error: ', np.mean(relative_error))

            return relative_error, error


def evaluate_net(type, net_params, nft):
    logging.info("=========================================================================")

    data = io.loadmat('./rus_data/chemical_data_file.mat')  # 文件存放恩
    X = data['Inputs']  # 打开matlab中的文件
    Y = data['Targets']  # 打开matlab中的文件

    X = np.transpose(X)
    Y = np.transpose(Y)
    X, Y = shuffle_data(X, Y)

    dnn_reg = DnnRegression(model='chemical', batch_size=32)

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    global LNodes
    global af_type

    LNodes = net_params
    af_type = nft

    logging.info("网络参数：")
    logging.info(LNodes)
    logging.info("激活函数：%s" % af_type)

    error = None
    relative_error = None
    fold_num = 5
    for i in range(fold_num):
        print("-------------------------------> Round : (%d/%d)" % (i+1, fold_num))
        dnn_reg.load_data(X, Y, fold=fold_num, k=i)
        b_time = time.time()
        loss = dnn_reg.train_net(4000, skip_n=3000)
        logging.info('lost: %f' % loss)
        re, e = dnn_reg.test_net()

        print("MSE: %f, RMSE: %f" % (np.mean(e * e), np.sqrt(np.mean(e * e))))
        logging.info("MSE: %f, RMSE: %f" % (np.mean(e * e), np.sqrt(np.mean(e * e))))
        logging.info("relative_error: %f +- %f" % (np.mean(re), np.sqrt(np.var(re))))

        print("-------------------------------> run time : %d" % (time.time() - b_time))
        if relative_error is None:
            relative_error = re
            error = e
        else:
            relative_error = np.vstack((relative_error, re))
            error = np.vstack((error, e))

    logging.info("Total:")
    logging.info("MSE: %f, RMSE: %f" % (np.mean(error * error), np.sqrt(np.mean(error * error))))
    logging.info("relative_error: %f +- %f" % (np.mean(relative_error), np.sqrt(np.var(relative_error))))
    np.save("./rus_data/re_%s_l%d_n%d.npy" % (type, len(LNodes), LNodes[0]), relative_error)


if __name__ == '__main__':
    logging.basicConfig(filename='./dnn_reg_results.log', level=logging.INFO, format='%(asctime)s  %(message)s')
    logging.info("")
    # logging.info(">>>>>>>>>>网络每一层的输出拼接网络的输入作为下一层的输入")
    logging.info(">>>>>>>>>>传统网络")
    evaluate_net('PZT8', [32, 32, 32], 'tanh')


