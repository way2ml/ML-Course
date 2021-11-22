# 反向传播

单个训练样本的损失函数:
$$
\text{C}(\mathbf{w}, \mathbf{b}) = \frac{1}{2}\| \mathbf{y} - \mathbf{o}\|^2=\frac{1}{2}\sum_{i=0}^{1}(\mathbf{y}_i - \mathbf{o}_i)^2
$$

损失函数 $\text{MSE}$:
$$
\text{L}(\mathbf{w}, \mathbf{b}) = \sum_{n=1}^N \text{C}^n(\mathbf{w}, \mathbf{b})
$$

对每一个训练样本得到的梯度分量求和，即可得到总的梯度分量: 
$$
\frac{\partial \text{L}(\mathbf{w}, \mathbf{b})}{\partial w} = \sum_{n=1}^N \frac{\partial \text{C}^n(\mathbf{w}, \mathbf{b})}{\partial w} \\
\frac{\partial \text{L}(\mathbf{w}, \mathbf{b})}{\partial b} = \sum_{n=1}^N \frac{\partial \text{C}^n(\mathbf{w}, \mathbf{b})}{\partial b}
$$


损失函数对第 $i=1$层的其中一个权重$w_{10}$的偏导数为:
$$
\frac{\partial C}{\partial w_{00}} = \frac{\partial C}{\partial z_{10}} \frac{\partial z_{10}}{\partial w_{00}}
$$

<img src='https://cdn.jsdelivr.net/gh/HuangJiaLian/DataBase0@master/uPic/2021_11_18_22_nn.png' width='40%'/>



## 前向传播计算$\frac{\partial z}{\partial w}$

由于激活$a_{10} = \sigma(z_{10})$ 的中间变量 $z_{10} = w_{00}a_{00} + w_{01}a_{01} + b_{10}$ ， 因此有$\frac{\partial z}{\partial w_{00}} = a_{00}$; 于是可以得到这一层中连接到该神经元的权重$w$对应的$\frac{\partial z}{\partial w}$ 分别为$a_{00}, a_{01}$ ; 那么整个第$i$层所有权重
$$
\left[\begin{array}{llll}
w_{00} & w_{01} \\ 
w_{10} & w_{11} \\ 
w_{20} & w_{21} \\ 
\end{array}\right]
$$


对应的$\frac{\partial z}{\partial w}$ 为, 每一组(行)对应的值是上一层的激活，每一行都是一样的， 这些值通过前向传播直接就算出来了:
$$
\left[ \begin{array}{llll}
a_{00} & a_{01}  \\ 
a_{00} & a_{01}  \\ 
a_{00} & a_{01}  \\ 
\end{array} \right]
$$


## 反向传播计算$\frac{\partial C}{\partial z}$

<img src='https://cdn.jsdelivr.net/gh/HuangJiaLian/DataBase0@master/uPic/2021_11_18_22_nn.png' width='40%'/>
$$
\frac{\partial C}{\partial z_{10}} = \frac{\partial C}{\partial a_{10}} \frac{\partial a_{10}}{\partial z_{10}}
$$

因为$a_{10} = \sigma (z_{10})$, 因此

$$
\frac{\partial a_{10}}{\partial z_{10}} = \sigma^{\prime}(z_{10})
$$
那么这一层所有节点对应的 $\frac{\partial a}{\partial z}$ 构成向量为 
$$
[\sigma^{\prime}(z_{10}), \sigma^{\prime}(z_{11}), \sigma^{\prime}(z_{12})]^{^\intercal}
$$
$z$ 在前向传播的时候可以记录下来，而 $\sigma ^{\prime}(z)$ 也是知道的。



还剩下一项:
$$
\frac{\partial C}{\partial a_{10}} = \frac{\partial C}{\partial z_{20}} \frac{\partial z_{20}}{\partial a_{10}} + \frac{\partial C}{\partial z_{21}} \frac{\partial z_{21}}{\partial a_{10}}
$$
<img src='https://cdn.jsdelivr.net/gh/HuangJiaLian/DataBase0@master/uPic/2021_11_18_22_nn.png' width='40%'/>

由于 $z_{20} = w_{00}a_{10} + w_{01}a_{11} + w_{02}a_{12}$ , 因此 $\frac{\partial z_{20}}{\partial a_{10}} = w_{00}$,  同理$\frac{\partial z_{21}}{\partial a_{10}} = w_{10}$, 于是上式变成:
$$
\frac{\partial C}{\partial a_{10}} = \frac{\partial C}{\partial z_{20}} w_{00} + \frac{\partial C}{\partial z_{21}} w_{10}
$$
同理，这一层其他神经元对应有
$$
\frac{\partial C}{\partial a_{11}} = \frac{\partial C}{\partial z_{20}} w_{01} + \frac{\partial C}{\partial z_{21}} w_{11} \\
\frac{\partial C}{\partial a_{12}} = \frac{\partial C}{\partial z_{20}} w_{02} + \frac{\partial C}{\partial z_{21}} w_{12}
$$
写成矩阵的形式是这样的
$$
\left[\begin{array}{l}
\frac{\partial C}{\partial a_{10}} \\ 
\frac{\partial C}{\partial a_{11}} \\ 
\frac{\partial C}{\partial a_{12}}\end{array}\right]
=

\left[\begin{array}{ll}
w_{00} & w_{10} \\ 
w_{01} & w_{11} \\ 
w_{02} & w_{12}\end{array}\right]

\left[\begin{array}{l}
\frac{\partial C}{\partial z_{20}} \\ 
\frac{\partial c}{\partial z_{21}}\end{array}\right]
$$
注意到
$$
\left[\begin{array}{ll}
w_{00} & w_{10} \\ 
w_{01} & w_{11} \\ 
w_{02} & w_{12}\end{array}\right] = 
\left[\begin{array}{lll}
w_{00} & w_{01} & w_{02}\\ 
w_{10} & w_{11} & w_{12}\end{array}\right]^{\intercal}
$$
这个关系会给我门带来很大的方便。

回到
$$
\frac{\partial C}{\partial z_{10}} = \frac{\partial C}{\partial a_{10}} \frac{\partial a_{10}}{\partial z_{10}} \\
= \sigma^{\prime}(z_{10}) \frac{\partial C}{\partial a_{10}} \\
= \sigma^{\prime}(z_{10})\left(w_{00}\frac{\partial C}{\partial z_{20}} + w_{10} \frac{\partial C}{\partial z_{21}}\right)
$$
再回到
$$
\frac{\partial C}{\partial w_{00}} = \frac{\partial C}{\partial z_{10}} \frac{\partial z_{10}}{\partial w_{00}} \\
= a_{00}\frac{\partial C}{\partial z_{10}} \\
= a_{00} \sigma^{\prime}(z_{10})\left(w_{00}\frac{\partial C}{\partial z_{20}} + w_{10} \frac{\partial C}{\partial z_{21}}\right)
$$

$$
\frac{\partial C}{\partial w_{00}^{\text{i}}} = a_{00} \sigma^{\prime}(z_{10})\left(w_{00}^{\text{j}}\frac{\partial C}{\partial z_{20}} + w_{10}^{\text{j}} \frac{\partial C}{\partial z_{21}}\right)
$$

汇总一下:
$$
\nabla \mathbf{w}^i = 
\left[\begin{array}{ll}
a_{00} & a_{01} \\ 
a_{00} & a_{01} \\ 
a_{00} & a_{01}\end{array}\right]

\odot

\left[\begin{array}{ll}
\frac{\partial C}{\partial z_{10}}  \\ 
\frac{\partial C}{\partial z_{11}}  \\ 
\frac{\partial C}{\partial z_{12}} \end{array}\right]
$$
其中,
$$
\left[\begin{array}{ll}
\frac{\partial C}{\partial z_{10}}  \\ 
\frac{\partial C}{\partial z_{11}}  \\ 
\frac{\partial C}{\partial z_{12}} \end{array}\right]

= 

\left[\begin{array}{ll}
\sigma^{\prime}(z_{10})  \\ 
\sigma^{\prime}(z_{11})  \\ 
\sigma^{\prime}(z_{12}) \end{array}\right]

\odot

\left[\begin{array}{l}
\frac{\partial C}{\partial a_{10}} \\ 
\frac{\partial C}{\partial a_{11}} \\ 
\frac{\partial C}{\partial a_{12}}\end{array}\right]

\\= 

\left[\begin{array}{ll}
\sigma^{\prime}(z_{10})  \\ 
\sigma^{\prime}(z_{11})  \\ 
\sigma^{\prime}(z_{12}) \end{array}\right]

\odot
\left(
\left[\begin{array}{ll}
w_{00} & w_{10} \\ 
w_{01} & w_{11} \\ 
w_{02} & w_{12}\end{array}\right]

\left[\begin{array}{l}
\frac{\partial C}{\partial z_{20}} \\ 
\frac{\partial c}{\partial z_{21}}\end{array}\right] \\
\right) 

\\=

\left[\begin{array}{ll}
\sigma^{\prime}(z_{10})  \\ 
\sigma^{\prime}(z_{11})  \\ 
\sigma^{\prime}(z_{12}) \end{array}\right]

\odot
\left(
\left[\begin{array}{lll}
w_{00} & w_{01} & w_{02}\\ 
w_{10} & w_{11} & w_{12}\end{array}\right]^{\intercal}

\left[\begin{array}{l}
\frac{\partial C}{\partial z_{20}} \\ 
\frac{\partial c}{\partial z_{21}}\end{array}\right] \\
\right)

\\=

\left[\begin{array}{ll}
\sigma^{\prime}(z_{10})  \\ 
\sigma^{\prime}(z_{11})  \\ 
\sigma^{\prime}(z_{12}) \end{array}\right]

\odot
\left(
{\mathbf{w}^{j}}^{\intercal}

\left[\begin{array}{l}
\frac{\partial C}{\partial z_{20}} \\ 
\frac{\partial c}{\partial z_{21}}\end{array}\right] \\
\right)
$$
即
$$
\frac{\partial C}{\partial \mathbf{z_1}} = \sigma^{\prime}(\mathbf{z_1}) \odot \left (\mathbf{w^2}^{\intercal} \frac{\partial C}{\partial \mathbf{z_2}} \right )
$$


于是:
$$
\nabla \mathbf{w}^1 = 
\left[\begin{array}{ll}
a_{00} & a_{01} \\ 
a_{00} & a_{01} \\ 
a_{00} & a_{01}\end{array}\right]

\odot

\left[\begin{array}{ll}
\frac{\partial C}{\partial z_{10}}  \\ 
\frac{\partial C}{\partial z_{11}}  \\ 
\frac{\partial C}{\partial z_{12}} \end{array}\right]

=

\left[\begin{array}{ll}
a_{00} & a_{01} \\ 
a_{00} & a_{01} \\ 
a_{00} & a_{01}\end{array}\right]

\odot

\left \{
\left[\begin{array}{ll}
\sigma^{\prime}(z_{10})  \\ 
\sigma^{\prime}(z_{11})  \\ 
\sigma^{\prime}(z_{12}) \end{array}\right]

\odot
\left(
{\mathbf{w}^{2}}^{\intercal}

\left[\begin{array}{l}
\frac{\partial C}{\partial z_{20}} \\ 
\frac{\partial C}{\partial z_{21}}\end{array}\right] \\
\right) 
\right \}
$$
即只要算出每个神经元的 $\frac{\partial c}{\partial z}$ 即可算出梯度的改变量， 先把最后一层的$\frac{\partial c}{\partial z}$算出，一层层就可往前算出了。

<img src='https://cdn.jsdelivr.net/gh/HuangJiaLian/DataBase0@master/uPic/2021_11_18_22_nn.png' width='40%'/>

输出层，
$$
\text{C}(\mathbf{w}, \mathbf{b}) = \frac{1}{2}\| \mathbf{y} - \mathbf{o}\|^2=\frac{1}{2}\sum_{i=0}^{1}(\mathbf{y}_i - \mathbf{o}_i)^2 = \frac{1}{2} \left( y_0^2 - 2y_0o_0 + o_0^2 + y_1^2 - 2y_1o_1 + o_1^2\right) 
$$

$$
o_0 = \sigma(z_{30})
$$


$$
\frac{\partial C }{\partial z_{30}} = \frac{\partial C}{\partial o_0} \frac{\partial o_0}{\partial z_{30}} = \left( o_0 - y_0 \right) \sigma^{\prime}(z_{30})
$$

同理
$$
\frac{\partial C }{\partial z_{31}} = \frac{\partial C}{\partial o_1} \frac{\partial o_1}{\partial z_{31}} = \left( o_1 - y_1 \right) \sigma^{\prime}(z_{31})
$$

$$
\frac{\partial C}{\partial \mathbf{z_{3}}} = (\mathbf{o} - \mathbf{y})\sigma^{\prime}(\mathbf {z_{3}})
$$



## 总结

相邻两层i, j关键量的计算关系: 
$$
\frac{\partial C}{\partial \mathbf{z_i}} = \sigma^{\prime}(\mathbf{z_i}) \odot \left (\mathbf{w^j}^{\intercal} \frac{\partial C}{\partial \mathbf{z_j}} \right )
$$
最后一层若j=o, 即第j层是输出层:
$$
\frac{\partial C}{\partial \mathbf{z_{o}}} = (\mathbf{o} - \mathbf{y})\sigma^{\prime}(\mathbf {z_{o}})
$$
第i层w的改变量:
$$
\nabla \mathbf{w}^i =  [\mathbf{a}_h,\mathbf{a}_h, \mathbf{a}_h]^{\intercal} \odot \frac{\partial C}{\partial \mathbf {z}_i}
$$


# 对于偏置

仿照类似的方法可以得到:
$$
\frac{\partial C}{\partial b_{10}} = \frac{\partial C}{\partial z_{10}} \frac{\partial z_{10}}{\partial b_{10}}
$$

## 前向传播计算$\frac{\partial z}{\partial b}$

$a_{10} = \sigma(z_{10})$ ;  $z_{10} = w_{00}a_{00} + w_{01}a_{01} + b_{10}$；因此, $\frac{\partial z_{10}}{\partial b_{10}} = 1$

那么,
$$
\frac{\partial \mathbf{z}_{1}}{\partial \mathbf{b}_{1}} = [1,1,1]^{\intercal}
$$

## 反向传播计算$\frac{\partial C}{\partial z}$

$$
\nabla \mathbf{b}^i =  \frac{\partial C}{\partial \mathbf {z}_i}
$$
