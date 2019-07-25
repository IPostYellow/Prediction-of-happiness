import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import pandas as pd

#创建变量
def create_placeholders(n_x, n_y):
    '''
    输入：n_x是X的维数
          n_y是Y的维数
    输出：X,Y是X和Y变量
    '''
    X = tf.compat.v1.placeholder(tf.compat.v1.float32,
                                 shape=[n_x, None],
                                 name='X')
    Y = tf.compat.v1.placeholder(tf.compat.v1.float32,
                                 shape=[n_y, None],
                                 name='Y')

    return X, Y


# 对参数进行初始化操作
def initialize_parameters(x_n, lay1_n, lay2_n, lay3_n):
    '''
    输入：lay1_n,lay2_n,lay3_n为每层的神经元个数,x_n为x的维数
    输出：parameters,其中包含初始化好后的三层神经网络的参数W1,b1,W2,b2,W3,b3
    '''
    tf.compat.v1.set_random_seed(1)

    W1 = tf.compat.v1.get_variable("W1", [lay1_n, x_n], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.compat.v1.get_variable("b1", [lay1_n, 1], initializer=tf.compat.v1.zeros_initializer())

    W2 = tf.compat.v1.get_variable("W2", [lay2_n, lay1_n], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.compat.v1.get_variable("b2", [lay2_n, 1], initializer=tf.compat.v1.zeros_initializer())

    W3 = tf.compat.v1.get_variable("W3", [lay3_n, lay2_n], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.compat.v1.get_variable("b3", [lay3_n, 1], initializer=tf.compat.v1.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    return parameters


# logistic回归实现向前传播，使用relu函数
def forward_propagation(X, parameters):
    '''
    输入：样本集X，参数集parameters(包含W1,W2,W3,b1,b2,b3)
    输出：Z3，神经网络第三层输出单元结果
    '''
    N = len(parameters) // 2

    for i in range(N):
        if i == 0:
            Zi = tf.add(tf.matmul(parameters["W1"], X), parameters["b1"])
            Ai = tf.nn.relu(Zi)
        else:
            Zi = tf.add(tf.matmul(parameters["W" + str(i + 1)], Ai), parameters["b" + str(i + 1)])
            Ai = tf.nn.relu(Zi)

    return Zi


# 成本函数定义
def compute_cost(Z3, Y):
    '''
    输入：神经网络最后输出的结果Z3，样本标签Y
    输出：计算好的成本值cost
    '''
    temp = tf.square(Z3 - Y)
    cost = tf.reduce_mean(temp)

    return cost


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):

    m = X.shape[1]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

#开始写神经网络
def DNN_happy(X_train, Y_train, X_test, Y_test,lay1_n,lay2_n,lay3_n,learning_rate=0.0001, num_epochs=1500, minibatch_size=32, print_cost=True):
    tf.set_random_seed(1)
    seed=3
    (n_x,m)=X_train.shape
    n_y=Y_train.shape[0]
    costs=[]
    #定义X,Y变量和形状
    X,Y=create_placeholders(n_x,n_y)
    #初始化参数
    parameters=initialize_parameters(n_x,lay1_n,lay2_n,lay3_n)
    #前向传播
    Z3=forward_propagation(X,parameters)
    #计算cost
    cost=compute_cost(Z3,Y)
    #反向传播使用Adam优化
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # Initialize all the variables
    init=tf.global_variables_initializer()

    #接下来要开始训练模型了，使用minibatch

    with tf.compat.v1.Session() as sess:
        sess.run(init)
    #训练开始
        for epoch in range(num_epochs):
            epoch_cost=0.
            num_minibatches=int(m/minibatch_size)#有多少个minibatches
            seed=seed+1
            minibatches=random_mini_batches(X_train,Y_train,minibatch_size,seed)#随机梯度下降
            for minibatch in minibatches:
                (minibatch_X,minibatch_Y)=minibatch
                _,minibatch_cost=sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
                epoch_cost+=minibatch_cost/num_minibatches#相当于把每个cost算出来然后再求平均，得到所有样本的cost
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:#训练五次就把cost记录下来
                costs.append(epoch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        print("Parameters have been trained!")
        #计算训练集准确率
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        #计算测试集准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters

train = pd.read_csv(r"C:\Users\黄寄\Desktop\天池新人赛\data\happiness_train_abbr.csv")
test = pd.read_csv(r"C:\Users\黄寄\Desktop\天池新人赛\data\happiness_test_abbr.csv")

X_train = train.drop(columns=['id','happiness','work_status','work_yr','work_type','work_manage'])
X_train.fillna(X_train.mean()['family_income'])

Y_train = train["happiness"]
Y_train = Y_train.map(lambda x:3 if x== -8 else x)

X_test = test.drop(columns=['id','happiness','work_status','work_yr','work_type','work_manage'])
X_test.fillna(X_test.mean()['family_income'])

Y_test = test["happiness"]
Y_test = Y_test.map(lambda x:3 if x== -8 else x)


parameters=DNN_happy(X_train,Y_train,X_test,Y_test,40,25,12,1)
print(parameters)
