import numpy as np


class network:
    # 神经网络的基本属性
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(self.sizes)
        self.weights = [np.random.normal(size=(self.sizes[i], self.sizes[i-1])) for i in range(1, self.num_layers)]
        self.biases = [np.random.normal(size=self.sizes[i]) for i in range(1, self.num_layers)]
    
    # 前向传播计算输出
    def out(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.activation_f(np.dot(w, a) + b)
        return a
    
    # 激活函数
    def activation_f(self, z):
        return 1.0/(1.0+np.exp(-z))
    
    # 激活函数的倒数 (反向传播的时候会用到)
    def activation_f_prime(self, z):
        return self.activation_f(z)*(1-self.activation_f(z))
    
    # 损失函数
    def loss(self, xs, ys):
        num_samples = len(xs)
        loss = 0
        for x, y in zip(xs, ys):
            out = self.out(x)
            v = y - out 
            loss = loss + v.dot(v)
        loss = loss / (2*num_samples)
        return loss
    
    # 准确度， 其中标签ys是one-hot格式
    def acc(self, xs, ys):
        num = xs.shape[0]
        outs = [np.argmax(self.out(x)) for x in xs]
        ys = [np.argmax(y) for y in ys]
        correct_num = sum(int(out == y) for out, y in zip(outs, ys))
        acc = correct_num / num
        return acc
    
    def backprop(self, x, y):
        # 目的: 找到要更新的量(要微调的量)
        nabla_ws = [np.zeros(w.shape) for w in self.weights]
        nabla_bs = [np.zeros(b.shape) for b in self.biases]
        
        partial_zs = [np.zeros(b.shape) for b in self.biases]
        # 前向传播得到每一层的加权求和值z, 与每一层的输出(激活)值a
        zs = []
        a = x
        a_s = [x]
        # 循环所有层
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.activation_f(z)
            zs.append(z)
            a_s.append(a)

        # 循环结束后的a即为网络的输出o
        o = a_s[-1]

        # 计算梯度
        # 对最后一层
        partial_z = (o - y)*self.activation_f_prime(zs[-1])
        partial_zs[-1] = partial_z

        nabla_ws[-1] =  (a_s[-2].reshape((-1, 1))* partial_zs[-1].reshape((1, -1))).T
        nabla_bs[-1] = partial_zs[-1]
        # 反向传播 
        # 输入层没有加权求和过激活等运算
        for n in range(1, self.num_layers -1): 
            h,  i,  j = -n-2,  -n-1,  -n
            partial_z = self.activation_f_prime(zs[i]) * np.dot(self.weights[j].T, partial_zs[j])
            partial_zs[i] = partial_z
            nabla_ws[i] =  (a_s[h].reshape((-1, 1))* partial_zs[i].reshape((1, -1))).T
            nabla_bs[i] = partial_zs[i]       
        return nabla_ws, nabla_bs
    
    # 更新参数
    def update(self, lr, xs, ys):
        
        # 训练样本的数目
        num_samples = xs.shape[0]
        
        # 目的: 找到要更新的量
        nabla_ws = [np.zeros(w.shape) for w in self.weights]
        nabla_bs = [np.zeros(b.shape) for b in self.biases]
        
        
        # 循环所有训练数据
        for x, y in zip(xs, ys):
            delta_nabla_ws, delta_nabla_bs = self.backprop(x,y)
            nabla_ws = [nw+dnw for nw, dnw in zip(nabla_ws, delta_nabla_ws)]
            nabla_bs= [nb+dnb for nb, dnb in zip(nabla_bs, delta_nabla_bs)]
        
        # 更新参数， 使得网络的性能变得更好
        self.weights = [weight - (lr/num_samples) * nabla_w for weight, nabla_w in zip(self.weights, nabla_ws)]
        self.biases = [biase - (lr/num_samples) * nabla_b  for biase,  nabla_b in zip(self.biases, nabla_bs)] 
        
    def sgd(self, xs, ys, lr, epochs, batch_size):
        assert xs.shape[0] == ys.shape[0]
        
        #  打乱顺序
        p = np.random.permutation(xs.shape[0])
        xs, ys = xs[p], ys[p]
        
        # 一大堆训练数据分成很多小份
        n = xs.shape[0]
        xs_batches = [ xs[k:k+batch_size] for k in range(0, n, batch_size)]
        ys_batches = [ ys[k:k+batch_size] for k in range(0, n, batch_size)]
        for epoch in range(epochs):
            for xs_batch, ys_batch in zip(xs_batches, ys_batches):
                # 用一小份数据去更新参数
                self.update(lr=lr, xs=xs_batch, ys=ys_batch)
            loss = self.loss(xs=xs, ys=ys)
            acc = self.acc(xs=xs, ys=ys)
            print('epoch: {:<5}\t loss: {:.4f}\t acc: {:.4f}'.format(epoch+1, loss, acc))
        

# 加载数据 得到一张图片的数据
import gzip
import pickle
f = gzip.open('mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
f.close()

xs, ys  = training_data[0], training_data[1]
def one_hot(ys):
    b = np.zeros((ys.size, ys.max()+1))
    b[np.arange(ys.size),ys] = 1
    return b 

ys =  one_hot(ys = ys)
xs_test, ys_test  = test_data[0], one_hot(ys=test_data[1])

net = network(sizes = [784, 30, 10])
net.sgd(xs=xs, ys=ys, lr=1, epochs=10, batch_size=30)
