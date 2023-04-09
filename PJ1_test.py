import idx2numpy
from matplotlib import pyplot as plt
from PIL import Image
from PJ1_model import MLP2_Runner
from PJ1_tool import cross_entropy, accuracy, Plot

# read the data
path = 'C:\\Users\\HJH\\Desktop\\CV\\PJ1\\'
X_test = idx2numpy.convert_from_file(path+'t10k-images.idx3-ubyte')
y_test = idx2numpy.convert_from_file(path+'t10k-labels.idx1-ubyte').reshape([-1,1])
img=Image.fromarray(X_test[0].astype('uint8'))
# show a test picture
plt.imshow(img,cmap='gray')
plt.show()
print(y_test[0])

X_test = X_test.reshape([-1,784])
X_test = X_test / 255

#  load the model and test
model_name = 'final_model.pkl'
runner = MLP2_Runner(2, cross_entropy, accuracy)
runner.load(model_name)

pred, test_score, test_loss = runner.predict(X_test, y_test, vis=1)
print("Test: accuracy = {:.2%}, loss = {:.4}".format(test_score, test_loss))

# -------------------------------------------------------------------------------------------
# # in this experiment, choose NO.7, NO.11, NO.13 set to build runner
# for i in [7, 11, 13]:
#     model_name = 'final' + str(i) + 'model.pkl'
#     runner = MLP2_Runner(2, cross_entropy, accuracy)
#     runner.load(model_name)
#
#     # test the model in test set
#     pred, test_score, test_loss = runner.predict(X_test, y_test, vis = 1)
#     print("Test: model {}, accuracy = {:.2%}, loss = {:.4}".format(i, test_score, test_loss))

