# Databricks notebook source
#%% 加载MNIST数据集
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# COMMAND ----------

print('train_images.shape:', train_images.shape)
print('train_labels.shape:', len(train_labels))
print('test_images.shape:', test_images.shape)
print('test_labels.shape:', len(test_labels))

# COMMAND ----------

#%% 网络架构
from keras import models, layers

# 创建一个顺序模型
network = models.Sequential()

# 添加一个全连接层，包含512个神经元，激活函数为ReLU，输入形状为28*28
network.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))

# 添加一个全连接层，包含10个神经元，激活函数为softmax
network.add(layers.Dense(10, activation='softmax'))

#%% 编译步骤
# 编译模型，使用rmsprop优化器，损失函数为categorical_crossentropy，评估指标为准确率
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
    metrics=['accuracy'])

#%% 准备图像数据
# 将训练图像和测试图像从28x28的二维数组转换为784的一维数组，并将像素值归一化到0-1之间
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# COMMAND ----------

#%% 准备标签
from keras.utils import to_categorical

# 将训练标签转换为one-hot编码
train_labels = to_categorical(train_labels)

# 将测试标签转换为one-hot编码
test_labels = to_categorical(test_labels)

# COMMAND ----------

#%% 拟合模型
# 训练模型，使用训练数据拟合模型，设置训练轮数为5，每批次训练样本数为128
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# COMMAND ----------

#%% 检测模型性能
test_loss, test_acc = network.evaluate(test_images, test_labels)

# COMMAND ----------

print('test_acc:', test_acc)

# COMMAND ----------

#%% 显示数据集中的图片
import matplotlib.pyplot as plt 
digit = train_images[4].reshape((28, 28))
plt.imshow(digit, cmap='gray')
plt.show()

# COMMAND ----------

#%% 使用函数式API定义相同模型
from keras import optimizers
# 定义输入张量，形状为28*28
input_tensor = layers.Input(shape=(28*28, ))
# 添加一个全连接层，包含32个神经元，激活函数为ReLU
x = layers.Dense(32, activation='relu')(input_tensor)
# 添加输出层，包含10个神经元，激活函数为softmax
output_tensor = layers.Dense(10, activation='softmax')(x)
# 创建模型，指定输入和输出
model = models.Model(inputs=input_tensor, outputs=output_tensor)
# 编译模型，指定优化器、损失函数和评估指标
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001), loss='mse', 
    metrics=['accuracy'])
# 训练模型，使用训练数据和标签，批量大小为128，训练10个周期
model.fit(train_images, train_labels, batch_size=128, epochs=10)
