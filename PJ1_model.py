import numpy as np
import pickle
from PJ1_tool import SGD, cross_entropy, accuracy, Plot, visualize

def relu(ipt):
    return np.maximum(0, ipt)

def softmax(ipt):
    ipt_max = np.max(ipt, axis=1).reshape([-1,1])
    ipt_exp = np.exp(ipt - ipt_max)
    partition = np.sum(ipt_exp, axis=1).reshape([-1,1])
    return ipt_exp / partition

class Linear:
    # Linear layer for MLP2
    def __init__(self, ipt_size, opt_size, lamda):
        """
        X.shape: [N (num of data), 784]
        Y = XW + b
        :param ipt_size: dim of input
        :param opt_size: dim of output
        """

        self.ipt_size = ipt_size
        self.opt_size = opt_size
        self.input = None
        self.lamda = lamda

        self.para = {}
        self.para['W']=np.random.randn(ipt_size,opt_size)
        self.para['b'] = np.zeros([1, opt_size])
        # grad: the gradient of linear layer
        self.grad = {}

    def __call__(self,ipt):
        return self.forward(ipt)

    def forward(self, ipt):
        """
        forward propagation
        :param ipt: input tensor, shape: N * 784
        :return: output tensor
        """

        self.input=ipt
        opt = np.matmul(ipt, self.para['W'])+self.para['b']
        return opt

    def backward(self, grad):
        """
        backward propagation
        :param grad: grad of Y
        :return: grad of X
        """

        self.grad['W'] = np.matmul(np.transpose(self.input), grad) + self.lamda * self.para['W']
        self.grad['b'] = np.sum(grad)

        return np.matmul(grad,np.transpose( self.grad['W']))

class ReLu:
    def __init__(self):
        self.input = None
        self.para = None

    def __call__(self,ipt):
        return self.forward(ipt)

    def forward(self, ipt):
        """
        forward propagation
        :param ipt: input tensor, shape: N * 784
        :return: output tensor after ReLu
        """

        self.input = ipt
        return relu(ipt)

    def backward(self, grad):
        """
        backward propagation
        :param grad: grad of Y
        :return: grad of X
        """

        return np.where(self.input <= 0, 0, grad)

class SoftmaxMultiLoss:
    def __init__(self, lamda):
        """
        :param ipt_size: dim of input
        :param opt_size: dim of output
        """
        self.input = None
        self.output = None
        self.para = None
        self.lamda = lamda

    def __call__(self, ipt):
        return self.forward(ipt)

    def forward(self, ipt):
        """
        forward propagation
        :param ipt: input tensor, shape: N * 784
        :return: output tensor
        """

        self.input = ipt
        self.output = softmax(ipt)
        return self.output

    def backward(self, pred, labels):
        """
        :param labels: true labels
        """
        N = pred.shape[0]
        one_hot_labels = np.zeros(pred.shape)
        for i in range(labels.shape[0]):
            tem = np.zeros([pred.shape[1]])
            tem[labels[i]] = 1
            one_hot_labels[i] = tem

        opt_grad = (pred - one_hot_labels) / N
        return opt_grad

class MlpModel:

    def __init__(self,ipt_size, hid_size, opt_size, lamda):
        self.lamda = lamda
        self.L1 = Linear(ipt_size, hid_size, self.lamda)
        self.A1 = ReLu()
        self.L2 = Linear(hid_size, opt_size, self.lamda)
        self.A2 = SoftmaxMultiLoss(lamda)
        self.layers = [self.L1, self.A1, self.L2, self.A2]

    def call(self, ipt):
        return self.forward(ipt)

    def forward(self, ipt, vis = 0):

        opt1 = self.L1(ipt)
        # print("opt1:", opt1[0][0])
        opt2 = self.A1(opt1)
        # print("opt2:", opt2[0][1])
        opt3 = self.L2(opt2)
        # print("opt3:", opt3[0][0])
        pred = self.A2(opt3)
        # print("pred:", pred[0][0])

        if vis == 1:
            visualize(self.L1.para['W'][:,0].reshape([28,28]), 'L1')
            visualize(self.L2.para['W'], 'L2')

        return pred

    def backward(self, pred, labels):
        grad1 = self.A2.backward(pred, labels)
        # print("grad1:",grad1[0][0])
        grad2 = self.L2.backward(grad1)
        # print("grad2:", grad2[0][0])
        grad3 = self.A1.backward(grad2)
        # print("grad3:", grad3[0])
        grad4 = self.L1.backward(grad3)
        # print("grad4:", grad4[0][0])

class MLP2_Runner(object):
    def __init__(self, flg, loss_func, metric, model = None, optimizer = None ):
        """

        :param flg: 1: train, 2: test
        :param loss_func: 损失函数
        :param metric: 模型评价方法
        :param model: 用于训练的模型架构
        :param optimizer: 参数优化器
        """

        if flg == 1:  # train
            self.model = model
            self.optimizer = optimizer
        self.train_loss = []
        self.train_score = []
        self.loss_func = loss_func
        self.metric = metric
        self.lr = 0.1

    def train(self, X_train, y_train, **kwargs):

        print_frq = kwargs.get("print_frq",100)
        num_epochs = kwargs.get("num_epochs", 1000)
        self.lr = kwargs.get("lr", 0.1)
        lamda = kwargs.get("lamda", 0.1)
        name = kwargs.get("name", "")
        plot_flg = kwargs.get("plot_flg", 0)
        best_score = 0.
        for epoch in range(num_epochs):

            state = np.random.get_state()
            np.random.shuffle(X_train)
            np.random.set_state(state)
            np.random.shuffle(y_train)

            pred = self.model.forward(X_train)
            train_score = self.metric(pred, y_train).item()
            train_loss = self.loss_func(self.model, pred, y_train, lamda).item()
            self.model.backward(pred, y_train)
            self.optimizer(self.model, self.lr)

            self.train_score.append(train_score)
            self.train_loss.append(train_loss)

            if train_score > best_score:
                with open(name + "model.pkl", 'wb') as f:
                    pickle.dump(self.model, f)
                best_score = train_score

            if epoch % print_frq == 0:
                print("Train: epoch = {}, accuracy = {:.4}, loss = {:.4}".format(epoch, train_score, train_loss))
                np.save(name + "train_score.py", np.asarray(self.train_score))
                np.save(name + "train_loss.py", np.asarray(self.train_loss))

        print("The best score of this train: {:.4}".format(best_score))
        if plot_flg:
            Plot(num_epochs, self.train_score, self.train_loss)

    def load(self, name):
            with open(name, "rb") as f:
                self.model = pickle.load(f)

    def predict(self, X_test, y_test, vis = 0):
        pred = self.model.forward(X_test, vis)
        test_score = self.metric(pred, y_test).item()
        test_loss = self.loss_func(self.model, pred, y_test, self.lr).item()
        return pred, test_score, test_loss

def super_para(X, y, r1, r2, r3, **kwargs):

    k =  kwargs.get("k", 0.8)
    num = X.shape[0]
    end = int(k * num)
    X_train = X[0:end,]
    y_train = y[0:end,]
    X_test = X[end + 1:num,]
    y_test = y[end + 1:num, ]

    ipt_size, opt_size = 784, 10
    optimizer = SGD
    loss_func = cross_entropy
    metric = accuracy
    print_frq = kwargs.get("print_frq", 50)
    num_epochs = kwargs.get("num_epochs",2000)

    max = 0.
    best_lr, best_hidsize, best_lamda = 0, 0, 0
    rst = []
    i = 0
    for lr in r1:
        for hid_size in r2:
            for lamda in r3:
                i += 1
                print("Search loop:",i)

                model = MlpModel(ipt_size, hid_size, opt_size, lamda)

                runner = MLP2_Runner(1, loss_func, metric, model, optimizer)
                runner.train(X_train, y_train, print_frq=print_frq, num_epochs=num_epochs,
                             lr=lr, lamda=lamda, name=str(i))

                runner.load(str(i)+"model.pkl")
                pred, score, loss = runner.predict(X_test, y_test)
                tem = [lr, hid_size, lamda, score, loss]
                rst.append(tem)
                if score > max:
                    best_lr = lr
                    best_hidsize = hid_size
                    best_lamda = lamda
                    max = score

    print("best score in train set: {:.4}".format(max))

    return best_lr, best_hidsize, best_lamda, rst


