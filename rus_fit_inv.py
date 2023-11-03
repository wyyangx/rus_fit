
import numpy as np
import tensorflow as tf
import os
import time
import logging


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
        # l = 0
        # net = input
        # for lu in LNodes:
        #     l += 1
        #     net = active_fun(bn(dense(net, lu, name="fc%d" % l, alpha=alpha),
        #                         is_training, name="bn%d" % l), name="a%d" % l, type=af_type)
        #
        # l += 1
        # net = dense(net, output_dim, name="fc%d_1" % l, alpha=alpha)

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


class RusFit:
    def __init__(self, alpha=0.0, batch_size=128):
        self.input_d = -1
        self.output_d = -1
        self.freq_index = []
        self.alpha = alpha
        self.batch_size = batch_size

        self.paras = np.empty([0, 1])
        self.freqs = np.empty([0, 1])
        self.paras_train = np.empty([0, 1])
        self.freqs_train = np.empty([0, 1])
        self.paras_valid = np.empty([0, 1])
        self.freqs_valid = np.empty([0, 1])
        self.paras_test = np.empty([0, 1])
        self.freqs_test = np.empty([0, 1])

        self.paras_mean = np.empty([0, 1])
        self.paras_var = np.empty([0, 1])
        self.freqs_mean = np.empty([0, 1])
        self.freqs_var = np.empty([0, 1])

        if not os.path.exists("./params"):
            os.mkdir("./params")
        self.model_name = "params/rus_fit_net"

    def load_data(self, data_path_list, fold, k):

        print("Load data ... ...")

        paras = np.empty([0, 1])
        freqs = np.empty([0, 1])

        for file in data_path_list:
            paras_file = file + '.paras.npy'
            tp = np.load(paras_file)
            if paras.shape[0] == 0:
                paras = tp
            else:
                paras = np.vstack((paras, tp))

            freqs_file = file + '.freqs.npy'
            tp = np.load(freqs_file)
            if freqs.shape[0] == 0:
                freqs = tp
            else:
                freqs = np.vstack((freqs, tp))

        freqs = freqs[0:57800, :]
        self.freq_index = []
        for i in range(freqs.shape[1]):
            self.freq_index.append(i)

        freqs = freqs[:, self.freq_index]
        print(freqs.shape)

        sample_index = range(freqs.shape[0])

        paras = paras[sample_index, :]
        freqs = freqs[sample_index, :]

        print(freqs.shape)

        self.paras = paras
        self.freqs = freqs

        xs, ys, self.freqs_test, self.paras_test = split_data(freqs, paras, fold, k, False)
        self.freqs_train, self.paras_train, self.freqs_valid, self.paras_valid = split_data(xs, ys, 5, 1, True)

        self.input_d = paras.shape[1]
        self.output_d = len(self.freq_index)

        # 求均值方差
        self.paras_mean = np.mean(self.paras_train, axis=0)
        self.paras_var = np.sqrt(np.var(self.paras_train, axis=0))

        self.freqs_mean = np.mean(self.freqs_train, axis=0)
        self.freqs_var = np.sqrt(np.var(self.freqs_train, axis=0))

        np.save(self.model_name + "_freqs_mean.npy", self.freqs_mean)  # 保存为.npy格式
        np.save(self.model_name + "_freqs_var.npy", self.freqs_var)  # 保存为.npy格式
        np.save(self.model_name + "_freq_index.npy", self.freq_index)  # 保存为.npy格式

        np.save(self.model_name + "_paras_mean.npy", self.paras_mean)  # 保存为.npy格式
        np.save(self.model_name + "_paras_var.npy", self.paras_var)  # 保存为.npy格式

        self.paras_train = self.paras_train - np.tile(self.paras_mean, (self.paras_train.shape[0], 1))
        self.paras_train = self.paras_train / np.tile(self.paras_var, (self.paras_train.shape[0], 1))
        self.freqs_train = self.freqs_train - np.tile(self.freqs_mean, (self.freqs_train.shape[0], 1))
        self.freqs_train = self.freqs_train / np.tile(self.freqs_var, (self.freqs_train.shape[0], 1))

        self.paras_valid = self.paras_valid - np.tile(self.paras_mean, (self.paras_valid.shape[0], 1))
        self.paras_valid = self.paras_valid / np.tile(self.paras_var, (self.paras_valid.shape[0], 1))
        self.freqs_valid = self.freqs_valid - np.tile(self.freqs_mean, (self.freqs_valid.shape[0], 1))
        self.freqs_valid = self.freqs_valid / np.tile(self.freqs_var, (self.freqs_valid.shape[0], 1))

        self.paras_test = self.paras_test - np.tile(self.paras_mean, (self.paras_test.shape[0], 1))
        self.paras_test = self.paras_test / np.tile(self.paras_var, (self.paras_test.shape[0], 1))
        self.freqs_test = self.freqs_test - np.tile(self.freqs_mean, (self.freqs_test.shape[0], 1))
        self.freqs_test = self.freqs_test / np.tile(self.freqs_var, (self.freqs_test.shape[0], 1))

        # self.paras_train = self.convert_paras(self.paras_train)
        # self.paras_test = self.convert_paras(self.paras_test)
        # self.input_d = self.paras_train.shape[1]

    @staticmethod
    def convert_paras(paras):
        n = paras.shape[1]
        to_paras = np.zeros([paras.shape[0], int(n*(n+1)/2+n)])
        for i in range(paras.shape[0]):
            para = paras[i, :]
            to_paras[i, 0:n] = para
            j = n
            for r in range(n):
                for c in range(r+1):
                    to_paras[i, j] = para[r] * para[c]
                    j = j + 1
        return to_paras

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

    def train_net(self, epochs):
        print("Train net ... ...")
        fit_op_train, fit_op_test, input_x, out_y, eval_loss, optimizer = self.deep_net()
        saver = tf.train.Saver()

        batch_size = self.batch_size

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            re = None
            min_loss = 1E5
            min_loss_epoch_n = 0
            for e in range(epochs):
                print('====>>>Epoch %i' % e)
                freqs_xs, paras_ys = shuffle_data(self.freqs_train, self.paras_train)
                for i in range(self.paras_train.shape[0] // batch_size):
                    beg_i = batch_size * i
                    end_i = batch_size * (i + 1)
                    bxs = paras_ys[beg_i:end_i, :]
                    bys = freqs_xs[beg_i:end_i, :]

                    # Run optimization op (backprop) and cost op (to get loss value)
                    sess.run([optimizer], feed_dict={input_x: bxs, out_y: bys})

                if (e + 1) < 10000:
                    continue

                if (e + 1) % 5 != 0:
                    continue

                # Validating
                loss_l = []
                for i in range(self.freqs_valid.shape[0] // batch_size):
                    beg_i = batch_size * i
                    end_i = batch_size * (i + 1)
                    bxs = self.paras_valid[beg_i:end_i, :]
                    bys = self.freqs_valid[beg_i:end_i, :]

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
            relative_error = np.empty([0, 200])

            for i in range(self.freqs_test.shape[0] // batch_size):
                beg_i = batch_size * i
                end_i = batch_size * (i + 1)
                bxs = self.paras_test[beg_i:end_i, :]
                bys = self.freqs_test[beg_i:end_i, :]
                #
                out_freqs = sess.run(fit_op_test, feed_dict={input_x: bxs})
                #
                out_freqs = out_freqs * self.freqs_var + self.freqs_mean
                real_freqs = bys * self.freqs_var + self.freqs_mean
                re = 100 * np.abs(real_freqs - out_freqs) / real_freqs
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

            return relative_error


def evaluate_net(type, net_params, nft):
    logging.info("=========================================================================")
    logging.info("材料：%s" % type)

    if type == 'LiNbO3':
        data_path_list = ['./rus_data/Notc2h_N02_20230602_0', './rus_data/Notc2h_N02_20230602_1',
                          './rus_data/Notc2h_N02_20230627_0', './rus_data/Notc2h_N02_20230627_1',
                          './rus_data/Notc2h_N02_20230627_2']
        run_fit = RusFit(batch_size=128)
    else:
        data_path_list = ['./rus_data/PZT8_20230706_0', './rus_data/PZT8_20230706_1',
                          './rus_data/PZT8_20230706_2', './rus_data/PZT8_20230706_3']
        run_fit = RusFit(batch_size=128)

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    global LNodes
    global af_type

    LNodes = net_params
    af_type = nft

    logging.info("网络参数：")
    logging.info(LNodes)
    logging.info("激活函数：%s" % af_type)

    relative_error = np.empty([0, 200])
    fold_num = 5
    for i in range(fold_num):
        print("-------------------------------> Round : (%d/%d)" % (i+1, fold_num))
        run_fit.load_data(data_path_list, fold=fold_num, k=i)
        b_time = time.time()
        loss = run_fit.train_net(12000)
        logging.info('lost: %f' % loss)
        re = run_fit.test_net()
        logging.info("relative_error: %f +- %f" % (np.mean(re), np.sqrt(np.var(re))))
        print("-------------------------------> run time : %d" % (time.time() - b_time))
        relative_error = np.vstack((relative_error, re))

    logging.info("relative_error: %f +- %f" % (np.mean(relative_error), np.sqrt(np.var(relative_error))))
    np.save("./rus_data/re_%s_l%d_n%d.npy" % (type, len(LNodes), LNodes[0]), relative_error)


def re_static():
    # re_LiNbO3_tanh_l6_n320
    re = np.load('./rus_data/re_LiNbO3_tanh_l6_n320.npy')
    re_st = np.zeros([6])
    count = 0

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
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
            re_st[index] += 1
            count += 1

    re_st /= count
    print(re_st)


if __name__ == '__main__':
    logging.basicConfig(filename='./results.log', level=logging.INFO, format='%(asctime)s  %(message)s')
    logging.info("")
    logging.info(">>>>>>>>>>网络每一层的输出拼接网络的输入作为下一层的输入")
    #
    # # evaluate_net('LiNbO3', [256, 256], 'lrelu')
    # # evaluate_net('LiNbO3', [256, 256, 256], 'lrelu')
    # # evaluate_net('LiNbO3', [256, 256, 256, 256], 'lrelu')
    # # evaluate_net('LiNbO3', [256, 256, 256, 256, 256], 'lrelu')
    # # evaluate_net('LiNbO3', [192, 192, 192], 'lrelu')
    # # evaluate_net('LiNbO3', [320, 320, 320], 'lrelu')
    # # evaluate_net('LiNbO3', [128, 128, 128], 'lrelu')
    # # evaluate_net('LiNbO3', [192, 192, 192], 'sigmoid')
    # # evaluate_net('LiNbO3', [192, 192, 192], 'tanh')
    # #
    # evaluate_net('PZT8', [256, 256], 'lrelu')
    # evaluate_net('PZT8', [256, 256, 256], 'lrelu')
    # evaluate_net('PZT8', [256, 256, 256, 256], 'lrelu')
    # evaluate_net('PZT8', [256, 256, 256, 256, 256], 'lrelu')
    # evaluate_net('PZT8', [192, 192, 192], 'lrelu')
    # evaluate_net('PZT8', [320, 320, 320], 'lrelu')
    # evaluate_net('PZT8', [128, 128, 128], 'lrelu')
    # evaluate_net('PZT8', [192, 192, 192], 'sigmoid')
    # evaluate_net('PZT8', [192, 192, 192], 'tanh')

    evaluate_net('PZT8', [256, 256, 256], 'sigmoid')

    # re_static()

    # np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    # out_freqs = np.load('./rus_data/out_freqs_PZT8_tanh_l6_n320.npy')
    # real_freqs = np.load('./rus_data/real_freqs_PZT8_tanh_l6_n320.npy')
    # relative = 100 * np.abs(real_freqs - out_freqs) / real_freqs
    # print(real_freqs[0, :])
    # print(out_freqs[0, :])
    # print(relative[0, :])
