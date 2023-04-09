import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def accuracy(pred, labels):
    pred_label = np.argmax(pred, axis = 1).reshape([-1,1])

    acc = np.sum(np.equal(pred_label,labels)) / labels.shape[0]
    return acc

def visualize(para, name):
    max = np.max(para)
    min = np.min(para)
    para = (para - min) / (max - min) * 255
    img=Image.fromarray(para.astype('uint8'))
    plt.imshow(img,cmap='gray')
    plt.title(name)
    plt.show()

def Plot(num_epoch, acc, loss):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    epochs = np.linspace(0, num_epoch, num_epoch, endpoint=False, dtype='int64')

    plt.plot(epochs, loss, color='#8E004D', label="Train loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(loc='upper right')
    plt.subplot(1, 2, 2)

    plt.plot(epochs, acc, color='#8E004D', label="Train accuracy")
    plt.ylabel("score")
    plt.xlabel("epoch")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("figure")
    plt.show()

def SGD(model, lr):
    for layer in model.layers:
        if isinstance(layer.para, dict):
            for key in layer.para.keys():
                layer.para[key] = layer.para[key] - lr * layer.grad[key]

def cross_entropy(model, pred, labels, lamda):
    num = pred.shape[0]
    loss = 0
    for i in range(0, num):
        idx = labels[i]
        loss -= np.log(pred[i][idx] + 1e-6)
    w = 0
    for layer in model.layers:
        if isinstance(layer.para, dict):
            w += np.linalg.norm(layer.para['W'])
    loss = loss / num + lamda / (2 * num) * w

    return loss
