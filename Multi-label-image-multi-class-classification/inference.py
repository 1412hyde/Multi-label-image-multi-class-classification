import numpy as np
import tensorflow as tf
import random

data     = np.load('mydata.npz')
trX, trY = data[ 'datax' ], data[ 'datay' ]

# shuffling the arrays
shuffling = list(zip(trX, trY))
random.shuffle(shuffling)
trX, trY = zip(*shuffling)
trX = np.asarray(trX)
trY = np.asarray(trY)

teX, teY = trX [ 3600:  ], trY [ 3600:  ]  # testset
trX, trY = trX [ :3600  ], trY [ :3600 ]  # trainset

trX = trX.reshape( -1, 32, 32, 1)
teX = teX.reshape( -1, 1024)
teY = teY.reshape( -1, 24)

class multinetwork(object):
    def __init__(self,x,y,lr):
        self.x = x
        self.y = y
        self.lr = lr
        self.opti = multinetwork.optimizer(self)
        self.cos = multinetwork.cost(self)
        self.acc1,self.acc2 = multinetwork.accuracy(self)
        self.scores,self.labels = multinetwork.test(self)


    def net(self):

        x = tf.reshape(self.x,shape=[-1,32,32,1])

        # w0 = tf.Variable(tf.random_normal([3, 3, 1, 32]))
        # b0 = tf.Variable(tf.random_normal([32]))
        # x_0 = tf.nn.relu(tf.nn.conv2d(x, w0, strides=[1, 1, 1, 1], padding="SAME") + b0)
        # p_0 = tf.nn.max_pool(x_0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


        w1 = tf.Variable(tf.random_normal([3,3,1,8]))
        b1 = tf.Variable(tf.random_normal([8]))
        x_1 = tf.nn.tanh(tf.nn.conv2d(x,w1,strides=[1,1,1,1],padding="SAME") + b1)
        p_1 = tf.nn.max_pool(x_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        w2 = tf.Variable(tf.random_normal([3,3,8,32]))
        b2 = tf.Variable(tf.random_normal([32]))
        x_2 = tf.nn.tanh(tf.nn.conv2d(p_1,w2,strides=[1,1,1,1],padding="SAME") + b2)
        p_2 = tf.nn.max_pool(x_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


        w_f1 = tf.Variable(tf.random_normal([8*8*32,128]))
        b_f1 = tf.Variable(tf.random_normal([128]))
        p_2f = tf.reshape(p_2,[-1,8*8*32])
        f_1 = tf.nn.tanh(tf.matmul(p_2f,w_f1) + b_f1)
        f_1d = tf.nn.dropout(f_1,0.5)

        w_f2 = tf.Variable(tf.random_normal([128,24]))
        b_f2 = tf.Variable(tf.random_normal([24]))
        prediction = tf.matmul(f_1d,w_f2) + b_f2

        tf.summary.histogram('pred', prediction)

        return prediction

    def cost(self):
        self.score = self.net()
        score_split = tf.split(self.score,8,1)
        label_split = tf.split(self.y,8,1)
        total = 0.0
        for i in range ( len(score_split)  ):
            total += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= score_split[i] ,labels= label_split[i] ))
        return total/8

    def optimizer(self):
        return tf.train.AdamOptimizer(learning_rate= self.lr).minimize(self.cost())

    def test(self):
        self.score = self.net()
        score_split = tf.split(self.score, 8, 1)
        label_split = tf.split(self.y, 8, 1)
        score_split = tf.round(tf.divide(tf.abs(score_split),1e+05,name=None),name=None)
        print score_split,label_split
        return  score_split,label_split

    def accuracy(self):
        self.score = self.net()
        score_split = tf.split(self.score, 8, 1)
        label_split = tf.split(self.y, 8, 1)

        correct_pred1 = tf.equal(tf.argmax(score_split[0], 1), tf.argmax(label_split[0], 1))
        correct_pred2 = tf.equal(tf.argmax(score_split[1], 1), tf.argmax(label_split[1], 1))

        return correct_pred1, correct_pred2

if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 32*32])
    y = tf.placeholder(tf.float32, [None, 24])
    lr = 0.1
    network = multinetwork(x,y,lr)
    batch_size = 200
    init = tf.global_variables_initializer()


    with tf.Session() as sess:
        sess.run(init)
        index = 0

        for batch_i in range(100):
            trData_i, trLabel_i = [], []

            trData_i .append( trX[ index : index + batch_size ] )
            trLabel_i.append( trY[ index : index + batch_size ] )
            index += batch_size
            if index > ( len(trX) - batch_size+1 ):
                index = 0
            trData_i = np.reshape(trData_i, (-1, 32 * 32))
            trLabel_i = np.reshape(trLabel_i, (-1, 24))


            sess.run(network.opti, feed_dict={x: trData_i, y: trLabel_i})
            if batch_i % 10 == 0:
                cost_tr = sess.run(network.cos, feed_dict={x: trData_i, y: trLabel_i})
                cost_te = sess.run(network.cos, feed_dict={x: teX[:3000], y: teY[:3000]})
                tf.summary.scalar('train_loss',cost_tr)
                tf.summary.scalar('test_loss',cost_te)
                # test accuracy
                accu1, accu2 = sess.run([network.acc1,network.acc2],
                                        feed_dict={x: teX[:3000], y: teY[:3000]})
                # print accu1,accu2
                sc,lb = sess.run([network.scores,network.labels],
                                        feed_dict={x: teX[:3000], y: teY[:3000]})


                numOfposit = 0.0
                for tt in range(accu1.shape[0]):
                    if accu1[tt] == accu2[tt] or accu1[tt] == True:
                        numOfposit += 1
                test_accu = numOfposit / accu1.shape[0]

                accu1, accu2 = sess.run([network.acc1,network.acc2],
                                        feed_dict={x: trData_i, y: trLabel_i})
                numOfposit = 0.0
                for tt in range(accu1.shape[0]):
                    if accu1[tt] == accu2[tt] or accu1[tt] == True:
                        numOfposit += 1
                train_accu = numOfposit / accu1.shape[0]
                print("%4d, cost_tr: %4.2g , cost_te: %4.2g , trainAccu: %4.2g , testAccu: %4.2g " % (
                batch_i, cost_tr, cost_te, train_accu, test_accu))

    writer = tf.summary.FileWriter('./logs', sess.graph)
    merge_op = tf.summary.merge_all()
