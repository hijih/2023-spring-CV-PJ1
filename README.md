# 2023-spring-CV-PJ1

## ①程序文件简介：
PJ1.train.py 用于模型训练的主文件
PJ1.test.py 用于模型测试的主文件
PJ1.tool.py 构建准确率计算与绘图、SGD、参数更新、交叉验证、参数可视化等函数
PJ1.model.py 构建两层神经网络模型

## ②训练步骤：
·打开PJ1.train.py文件
·将数据集所在位置传入“path”变量
·修改所需的超参数遍历范围，并分别传入“r1”~“r4”变量
·设置需要的训练轮数和打印频次
·若需要输出训练准确率与损失值变化的图像，则在调用MLP2_Runner.train时传入：plot_flg=1
·运行

## ③测试步骤
·打开PJ1.test.py文件
·将数据集所在位置传入“path”变量
·检查model的存储路径，保证正确导入
·若需要可视化每层参数，则在调用MLP2_Runner.predict时传入：vis=1
