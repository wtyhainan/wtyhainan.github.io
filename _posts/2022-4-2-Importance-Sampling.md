---
layout: "post"
title: "Importance Sampling"
date: 2022-4-2
categories: "Importance Sampling"
tags: "importance-sampling"
---

# Importance Sampling
&emsp; &ensp; 当我们在使用Monte Carlo方法来估算某一变量的具体取值时，Importance Sampling是一种常用来降低估计方差的方法。考虑如下问题：

$$\mathcal L = E_f \Big[H(\mathbf x) \Big ] = \int H(\mathbf x) \, f(\mathbf x) \,d \mathbf x \quad \quad (1)$$

这里 $H(\mathbf x)$ 为样本评价函数。一般情况下，可以使用Monte Carlo方法来估算 $\mathcal L$:

$$\mathcal L= \frac{1}{N} \sum_{i=1}^{N} H(\mathbf x_i) \quad \quad (2)$$

其中样本 $ x_i \sim f(x)$。现假设 $ g(x) $ 为另一个概率密度函数，改写式（1）， 有：

$$ 
\begin{aligned}
\mathcal L &= E_f \Big[ H(\mathbf x) \Big] = \int H(\mathbf x) \, f(\mathbf x) \, d \mathbf x \\
&= \int H(\mathbf x) \frac{f(\mathbf x)}{g(\mathbf x)} \, g(\mathbf x) \, d \mathbf x =E_g \Big[ H(\mathbf x) \frac{f(\mathbf x)}{g(\mathbf x)} \Big]
\end{aligned}
\quad(3)
$$

此时，我们可以从概率分布 $g(x)$ 中抽样来估计 $\mathcal L$。如下：

$$
\mathcal{ \hat L} = \frac{1}{N} \sum_{i=0}^N \frac{f(\mathbf x_i)}{g(\mathbf x_i)} \, H(\mathbf x_i) \quad \quad (4)
$$

&emsp; &ensp; 通常而言，我们可以选择任意的概率密度函数 $g(x)$ 通过（4）式来估算 $\mathcal L$ 的值。但在一些情况下，这并不是一个合理的做法。由于使用来自 $g(x)$ 的样本来估算 $\mathcal L$ 的值，考虑在样本服从 $g(x)$ 分布的情况下，通过使估计值 $\mathcal {\hat L}$ 的方差最小化来寻找最优的 $g^{ * }(x)$ 。

$$\mathop{min}\limits_{g} = Var_g \Big(  H(\mathbf X) \frac{f(\mathbf X)}{g(\mathbf X)} \Big) \quad \quad(5)$$

[文献] [1] 证明最优的 $g^{ * }(x)$ 为：

$$g^{ * }(x) = \frac{\mid H(\mathbf x) \mid f(\mathbf x)}{\int \mid H(\mathbf x) \mid f(\mathbf x) d \mathbf x} \quad \quad (6)$$

当 $H(\mathbf x) \geqslant 0$ 时，有 $g^{ * }(x) = \frac{H(\mathbf x) f(\mathbf x)}{ \mathcal L}$。

&emsp; &ensp; 想要通过（6）式来确定 $g^{ * }(x)$ 是很困难的，主要有两方面：第一是 $\mathcal L$ 未知 ***（这是我们想要估计的）***; 第二是 $H(\mathbf x)$ 有时候并不可知 ***(这里并不是特别理解)***。为了避开这两个困难，考虑通过样本 $ \Big ( H(\mathbf x_1), H(\mathbf x_2), ... ,H(\mathbf x_n) \Big )$ 来估计 $g^{ * }(x)$。

## 最小方差法

&emsp; &ensp; 一般情况下，我们会将 $g(x)$ 取为与 $f(x)$ 相同的概率分布族。现将 $g(x)$ 表示为 $f(x; v)$，原始的 $f(x)$ 表示为 $f(x;u)$，其中 $v$, $u$代表概率分布参数。在方差最小化准则下寻找最优参数 $v^{ * }$：

$$
\begin{aligned}
&
\mathop{min}\limits_{v \in \mathcal V} Var_v \Big( H(\mathbf X) W(\mathbf X; u, v) \Big ) \\
&= \mathop{min}\limits_{v \in \mathcal V} E_{f_v} \Big[ H^2(\mathbf X) W^2(\mathbf X; u, v) \Big] - E_{f_v} \Big[ H(\mathbf X) W(\mathbf X; u, v)\Big ]^2 \quad \quad(7)
\end{aligned}
$$ 

其中 $W(x; u, v) = \frac{f(\mathbf x;u)}{f(\mathbf x; v)} $。（7）式的第二项为常数，因此式（7）可以等价为：

$$ 
\begin{aligned} 
\mathop{min}\limits_{v \in \mathcal V} V(v) &= \mathop {min}\limits_{v \in \mathcal V} E_{f_v} \Big[  H^2(\mathbf X) W^2(\mathbf X;u, v) \Big] \\
&= \mathop {min}\limits_{v \in \mathcal V} E_{f_u} \Big[  H^2(\mathbf X) W(\mathbf X;u, v) \Big] \quad \quad (8)
\end{aligned}
$$

然后可以使用Monte Carlo方法来计算 $V(v)$，

$$\hat V = \frac{1}{N} \sum_{i=1}^N \Big[ H^2(\mathbf x_i) W(\mathbf x_i;u, v)\Big] \quad \quad (9) $$

$\mathbf x_1, ..., \mathbf x_n$ 是从概率分布 $f(x; u)$ 中采样得到。在一般的应用中，函数 $V(v)$ 和 $\hat V(v)$ 均是凸函数且可微，因此有：

$$ E_u \Big[ H^2(\mathbf X) \, \nabla_v W(\mathbf X; u, v) \Big] = 0 \quad \quad (10)$$

和

$$\frac{1}{N} \sum_{i=1}^N H^2(\mathbf x_i) \, \nabla_v W(\mathbf x_i; u, v) = 0 \quad \quad (11)$$

其中 

$$
\begin{aligned}
\nabla_v W(\mathbf x; u, v) &= \nabla_v \frac{f(\mathbf x;u)}{f(\mathbf x; v)} \\
&= \Big[ \nabla_v \ln f(\mathbf X; v)\Big] W(\mathbf X; u, v) \quad \quad (12)
\end{aligned}
$$

&ensp; &emsp; 求解式（11）可得到最优 $v{ * }$。

## Cross-Entropy方法

&emsp; &ensp; 另一个求解最优$g^{ * }(x)$的办法式Cross-Entropy。式（8）可写成如下等式：

$$
\begin{aligned}
\mathop{min}\limits_{v \in \mathcal V} V(v) &= \mathop{min}\limits_{v \in \mathcal V} E_v \Big[ H^2(\mathbf X) \frac{f^2(\mathbf X; u)}{f^2(\mathbf X; v)} \Big] \\
&= \mathop{min}\limits_{v \in \mathcal V}  E_w \Big[  H^2(\mathbf X) \frac{f^2(\mathbf X; u)}{f^2(\mathbf X; v)} \frac{f(\mathbf X; v)}{f(\mathbf X; w)}\Big] \\
&= \mathop{min}\limits_{v \in \mathcal V} E_w \Big[ H^2(\mathbf X) W(\mathbf X; u, v) W(\mathbf X; u, w)\Big] \quad \quad (13)
\end{aligned}
$$

其中 $w$ 为任意参数。对于$\mathbf x_1, \mathbf x_2, ... , \mathbf x_n \sim f(\mathbf x; w)$， 上式的Monte Carlo估计为：

$$
\mathop{min}\limits_{v \in \mathcal V} \hat V(v) =  \mathop{min}\limits_{v \in \mathcal V} \frac{1}{N} \sum_{i=1}^N H^2(\mathbf x_i) \, W(\mathbf x_i; u, v) \, W(\mathbf x_i; u, w)  \quad \quad(14)
$$

通过求解（14）式，可以的到最优 $v^{ * }$值。

&emsp; &ensp; KL散度常用来衡量两个概率分布之间的距离。两个概率分布 $g(x)$ 和 $f(x)$ 之间的KL散度定义为：

$$
\begin{aligned}
D_{KL}(f \mid \mid g) &= \int f(x) \ln \frac{f(x)}{g(x)} dx \\
&= \int f(x) \ln f(x) dx - \int f(x) \ln g(x) dx    \quad \quad(15)
\end{aligned}
$$

$D_{KL}$的第一项为为概率密度分布 $f(x)$ 的熵，当 $f(x)$ 固定式该值为一常数。因此最小化 $D_{KL}$ 等价于最大化交叉熵项 $\int f(x) \ln g(x) dx$,

$$\mathop{min}\limits_{g} D_{KL} \iff \mathop{max}\limits_g CE(g) \iff \mathop{max}\limits_{g} \int f(x) \ln g(x) dx  \quad \quad (16)$$

&emsp; &ensp; 当我们利用最小化交叉熵的方法来寻找最优 $g^{ * }(x)$ 时，通常会从 $g^{ * }(x)$ 的概率分布族中选择一个参数为 $v$ 的概率分布 $f(x; v)$ 。设 $H(\mathbf x) \ge 0$，因此最大化 $f(x;u)$ 和 $f(x; v)$ 之间的交叉熵距离等价最大化下式： 

$$ \mathop{max}\limits_{v \in \mathcal V}\int H(\mathbf x) f(\mathbf x; u) \ln f(\mathbf x; v) = \mathop{max}\limits_{v \in \mathcal V} E_{u} \Big[ H(\mathbf X) \ln f(\mathbf X; v) \Big] \quad \quad (17)$$

因为交叉熵函数为可微的凸函数，因此我们有：

$$
E_{u} = \Big[ H(\mathbf X) \nabla \ln f(\mathbf X; v)\Big] = 0 \quad \quad (18)
$$

和

$$\frac{1}{N} \sum_{i=1}^N H(\mathbf x_i) \nabla \ln f(\mathbf x_i; v) = 0 \quad \quad (19)$$

其中 $\mathbf x_1, ..., \mathbf x_n \sim f(x; u)$。最优 $v^{ * }$ 可通过解式（18）和（19）得到。引入一个参数为 $w$ 的分布 $f(x; w)$ ，式（17）可重写为：

$$\mathop{max}\limits_{v \in \mathcal V} E_w \Big[ H(\mathbf X) W(x; u, w) \ln f(\mathbf X; v) \Big] \quad \quad (20)$$

此时我们可以通过迭代的方式求解式（20）。

$$\mathop{max}\limits_v \hat {CE}(v) = \mathop{max}\limits_v \frac{1}{N} \sum_{i=1}^N H(\mathbf x_i) W(\mathbf x_i; u, w) \ln f(\mathbf x_i; v) \quad \quad (21)$$

其中 $\mathbf x_1, ... , \mathbf x_N \sim f(\mathbf x; w)$。我们可以不断地从分布 $f(\mathbf x; w)$ 中抽取样本并通过式（21）来优化参数 $v$，当满足$\hat {CE}(v)$ 收敛之后，则认为找到了最优参数 $v^{ * }$。

## 总结

&emsp; &ensp; 当我们使用Importance Sampling策略来解决Monte Calor近似时，会遇到如何选择抽样函数 $g(x)$ 地问题。在最小方差准则下，我们可以通过最小化方差的办法来得到最优 $g^{ * }(x)$ ，也可以通过Cross-Entropy的方法求解最优 $g^{ * }(x)$。直接最小化方差法要求我们求解式（11）这个非线性方程来得到最优 $g^{ * }(x)$。Cross-Entropy通过引入另一概率分布 $f(\mathbf x; w)$，不断地从 $f(\mathbf x; w)$ 抽取样本，迭代优化式（21）来得到最优 $g^ { * }(x)$。  


### 参考文献
[1] Simulation and the Monte Carlo method

[文献]: https://www.doc88.com/p-1764614416883.html?r=1

