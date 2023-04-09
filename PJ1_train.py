import numpy as np
from PJ1_model import MlpModel, MLP2_Runner, super_para
from PJ1_tool import SGD, cross_entropy, accuracy
import idx2numpy
from matplotlib import pyplot as plt

np.random.seed(42)

# read the data
path = 'C:\\Users\\HJH\\Desktop\\CV\\PJ1\\'
X_train = idx2numpy.convert_from_file(path+'train-images.idx3-ubyte').reshape([-1,784])
y_train = idx2numpy.convert_from_file(path+'train-labels.idx1-ubyte').reshape([-1,1])
y_train =y_train-1+1
X_train = X_train / 255

# show the distribution of labels
dt_ytrain={}
for x in y_train:
    dt_ytrain[x[0]]=dt_ytrain.get(x[0],0)+1
print("Labels of training data:",dt_ytrain)
plt.bar(list(dt_ytrain.keys()),list(dt_ytrain.values()),label="y_train",color='teal')
plt.xticks(list(dt_ytrain.keys()))
plt.xlabel('label name')
plt.ylabel('number')
plt.title('Labels of training data')
plt.show()

# shuffle the data
state = np.random.get_state()
np.random.shuffle(X_train)
np.random.set_state(state)
np.random.shuffle(y_train)

# the arrange of super parameter
r1 = np.logspace(-3, -1, 3)
r2 = np.linspace(100, 1000, num = 3, dtype='int64')
r3 = np.logspace(-3, -1, 2)

# find the best super parameter and save the information for further explore
lr, hid_size, lamda, rst = super_para(X_train, y_train, r1, r2, r3, num_epochs = 2000)
np.save("SuperPara.npy", rst)

ipt_size, opt_size = 784, 10
num_epochs = 5000
print_frq = 50

model = MlpModel(ipt_size, hid_size, opt_size, lamda)
optimizer = SGD
loss_func = cross_entropy
metric = accuracy

# use the best set of super parameter to build runner
runner = MLP2_Runner(1, loss_func, metric, model, optimizer)
runner.train(X_train, y_train, print_frq = print_frq, num_epochs = num_epochs,
            lr = lr, lamda =  lamda, name = "final_", plot_flg = 1)

# -------------------------------------------------------------------------------------------
# # in this experiment, choose NO.7, NO.11, NO.13 set to build runner
# i = 0
# for a in r1:
#     for b in r2:
#         for c in r3:
#             i+=1
#             if i in [7, 11, 13]:
#                 lr, hid_size, lamda =a, b, c
#                 model = MlpModel(ipt_size, hid_size, opt_size, lamda)
#                 runner = MLP2_Runner(1, loss_func, metric, model, optimizer)
#                 runner.train(X_train, y_train, print_frq = print_frq, num_epochs = num_epochs,
#                             lr = lr, lamda =  lamda, name = "Final"+str(i), plot_flg = 1)
