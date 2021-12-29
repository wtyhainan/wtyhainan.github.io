---
layout: "post"
title: stochastic policy gradient
subtitle: policy gradient
date: 2021-12-29
categories: "RL"
tags: "policy gradient" "stochastic policy gradient" 
---


# 随机策略梯度理论

### 1、策略梯度

**强化学习的目标**是找到一个最优策略，当agent遵循该策略与环境进行交互时收获尽可能多的回报。假设策略由参数 $\theta$ 表示，记为：$\pi(a|s;\theta)$，那么目标函数 $J$ 可以表示为:

$$
J(\theta)=\sum_{s \in S}d^{\pi}(s)V^{\pi}(s)=\sum_{s \in S}d^{\pi}(s)\sum_{a \in A}\pi(a|s;\theta)Q^{\pi}(s,a) \tag{1}
$$

为使用梯度算法找到最优$\theta$，需要计算 $J$ 对 $\theta$ 的梯度。在详细推到 $J$ 关于 $\theta$ 的梯度之前，这里给出 $J$ 关于参数 $\theta$ 梯度的理论表示式：

$$
\begin{aligned} 
\nabla_{\theta} J(\theta) &= \nabla \sum_{s \in S}d^{\pi}(s)V^{\pi}(s) 
\\ 
\\ 
&= \nabla_{\theta} \sum_{s \in S}d^{\pi}(s)\sum_{a \in A} \pi(a|s;\theta)Q^{\pi}(s,a)
\\ 
\\ 
 &= E_{\pi}( Q^{\pi}(s,a) \nabla_{\theta} log( \pi(a|s;\theta) ))  
\end{aligned}
\tag{2}
$$

### 2、详细推导过程

为方便表示 $\nabla_{\theta} J $，我们首先定义agent的一步转移概率:

$$
\rho(s \rightarrow s{'},k=1) = \sum \limits_{a \in A} \pi(a|s)p(s^{'}|s,a)
$$

显然，$ \rho(s \rightarrow s{'}, k=0)=1$ 。因此，agent从状态 $s$ 经过 $n$ 步达到状态 $x$ 是的概率，可以分为两步来计算，agent通过 $n-1$ 步到达中间状态$s^{'}$，再从状态 $s^{'}$ 经过一步转移到达状态 $x$ ：

$$
\rho (s \rightarrow x,k=n) = \sum\limits_{s^{'} \in S} \rho(s \rightarrow s^{'},k=n-1) \rho(s{'} \rightarrow x, k=1)
$$

这里首先采用下式计算$\nabla V(s)$，

$$
\begin{aligned}
\footnotesize \nabla_{\theta} V(s) &= \footnotesize \nabla_{\theta} \sum_{a \in A} \pi (a|s;\theta)Q^{\pi}(s,a)
\\
\\ &= \footnotesize \sum_{a \in A}(\nabla_{\theta} \pi(a|s;\theta)Q^{\pi}(s,a) + \pi(a|s;\theta) \nabla_{\theta} Q^{\pi}(s,a))
\\
\\ &= \footnotesize \sum_{a \in A} \nabla_{\theta} \pi(a|s; \theta)Q^{\pi}(s, a) + \sum_{a \in A} \pi(a|s;\theta) \nabla_{\theta} Q^{\pi}(s,a)
\\
\\ &= \footnotesize \Phi(s) + \sum_{a \in A} \pi(a|s;\theta) \nabla_{\theta} (\sum_{s^{'} \in S}P(s^{'},r|s,a)(r+V(s^{'})))
\\
\\ &= \footnotesize \Phi(s) + \sum_{a \in A} \pi(a|s;\theta) \nabla_{\theta} \sum_{s^{'} \in S}P(s^{'}|s,a)V(s^{'})
\\
\\ &= \footnotesize \Phi(s) + \sum_{a \in A} \pi(a|s;\theta) \sum_{s^{'} \in S}P(s^{'}|s,a) \nabla_{\theta} V(s^{'})  
\\
\\ &= \footnotesize \Phi(s) + \underline{ \sum_{s^{'} \in S} \rho (s \rightarrow s^{'}, k=1) \nabla_{\theta} V(s^{'})}
\\
\\ &= \footnotesize \Phi(s) + \sum\limits_{s^{'} \in S}    
\rho(s \rightarrow s^{'},k=1)   
\underline{(     \Phi(s^{'}) +   \sum_{s^{''} \in S} \rho (s^{'} \rightarrow s^{''}, k=1) \nabla_{\theta} V(s^{''}))}
\\
\\ &= \footnotesize \Phi(s) + \sum_{s^{'} \in S} \rho(s \rightarrow s^{'},k=1) \Phi(s^{'} )
+ \sum\limits_{s^{'} \in S} \rho(s \rightarrow s^{'}, k=1) 
\sum\limits_{s^{''} \in S} \rho(s^{'} \rightarrow s^{''}, k=1) \nabla_{\theta} V(s^{''})
\\
\\ &= \footnotesize \Phi(s) + \sum\limits_{s^{'} \in S} \rho(s \rightarrow s^{'}, k=1) \Phi(s^{'})
+ \sum\limits_{s^{''} \in S} \rho(s \rightarrow s^{''}, k=2) \Phi(s^{''}) +....
\\
\\ &= \footnotesize \sum\limits_{x \in S} \sum\limits_{k=1}^{\infty} \rho(s \rightarrow x, k) \Phi(x)
\end{aligned}
\tag{3}
$$

其中 $\Phi(s)= \sum\limits_{a \in A} \nabla_{\theta} \pi(a|s; \theta)Q^{\pi}(s,a)$  ，$P(s^{'},r|s,a)$ 表示agent在状态 $s$ 下执行动作 $a$ 后状态从 $s$ 转移到状态 $s^{'}$ 并得到及时奖励 $r$ 的概率。现在我们有：

$$
\nabla_{\theta} V(s) = \sum\limits_{x \in S} \sum\limits_{k=0}^{\infty} \rho(s \rightarrow x,k) \Phi(x) 
\tag{4}
$$

将（4）式代入（2）式得，

$$
\begin{aligned}
\nabla_{\theta} J(\theta) &= \nabla_{\theta} \sum\limits_{s \in S}d^{\pi}(s)V^{\pi}(s) 
\\
\\ &= \nabla_{\theta} V^{\pi}(s_0)
\qquad \qquad \qquad \qquad \qquad \qquad \scriptsize 假设agent从初始状态 s_{0}开始。
\\
\\ &= \sum\limits_{s \in S} \sum\limits_{k=0}^{\infty} \rho(s_0 \rightarrow s, k) \Phi(s) 
\qquad \qquad \qquad \scriptsize 令\sum_{k=0}^{\infty} \rho(s_0 \rightarrow s,k) = \eta(s)
\\
\\ &= \sum_{s \in S} \eta(s) \Phi(s)
\qquad \qquad \qquad \quad \qquad \qquad \scriptsize 归一化 \eta(s) \rightarrow \frac{\eta(s)}{\sum\limits_{s \in S} \eta(s)}
\\
\\ &= \sum\limits_{s \in S}\eta(s) \sum\limits_{s \in S} \frac{\eta(s)}{\sum\limits_{s \in S}\eta(s)}\Phi(s)
\qquad \qquad \qquad \scriptsize \sum\limits_{s \in S} \eta(s)=constant
\\
\\ &\approx \sum\limits_{s \in S} \frac{\eta(s)}{\sum\limits_{s \in S} \eta(s)} \Phi(s)=\sum\limits_{s \in S} d^{\pi}(s) \Phi(s)
\\
\\ &= \sum\limits_{s \in S} d^{\pi}(s) \sum\limits_{a \in A} \nabla_{\theta} \pi(a|s;\theta)Q^{\pi}(s,a)
\\
\\ &= E_{\pi}(\nabla_{\theta} log(\pi(a|s;\theta))Q^{\pi}(s,a))
\qquad \qquad \qquad \scriptsize E_{\pi}表示 E_{a \sim A,s \sim S}
\end{aligned}
\tag{5}
$$

至此，有关策略梯度的理论推导已经完成。由于我们是从 $s_0$ 状态出发，推导 $J(\theta)$ 关于参数 $\theta$ 的梯度，这就导致策略梯度属于on-policy方法，对于智能体agent的训练只能基于当前policy采样得到的样本来进行。

### 3、总结

1）策略梯度属于*on-policy*方法，训练数据只能基于当前policy与环境交互来获得；

2）从 $\nabla_{\theta} J = E_{\pi}( \nabla_{\theta}log(\pi(a|s;\theta))Q^{\pi}(s,a))$ 中看出，当agent在状态 $s$ 下根据策略 $ \pi(a|s;\theta)$ 采取行动 $a$ 与环境交互时，$Q^{\pi}(s,a)$ 给出了动作 $a$ 的优良性评价，策略梯度根据该评价适当的增加（或减少）agent在状态 $s$ 下选择动作 $a$ 的概率；

3）由式（5）式可知，对于不同episode， $Q^{\pi}(s,a)$ 对真实值估计的方差很大，这经常导致训练不稳定。假设agent在状态 $s$ 下可 $a_0, a_1, a_2$ 三个动作，每个动作对应的 $Q$ 值分别为 $Q(s,a_0)=10, Q(s,a_1)=8, Q(s,a_2)=-0.1$ ，此时由策略梯度理论，策略梯度算法会向 $\nabla_{\theta}log(\pi(a_0|s;\theta))$ 和 $\nabla_{\theta}log(\pi(a_1|s;\theta))$ 正方向及 $\nabla_{\theta}log(\pi(a_2|s;\theta))$ 相反方向移动。如下图：

<img title="" src="file:///C:/Users/yangyuanbao/Desktop/my%20post/正确策略梯度示意图.jpg" alt="正确梯度示意图" data-align="center" width="471">

此时策略梯度算法能很好的区分不同动作的优劣性，算法能向折扣回报高的动作移动的同时，远离折扣回报低的动作。然而，在梯度方向相同，$Q$ 值变为：$Q^{\pi}(s,a_0)=10.2,Q^{\pi}(s,a_1)=8.2,Q^{\pi}(s,a_2)=0.1$ 时，由于3个动作的Q值均大于0，策略梯度算法会同时朝着这3个动作的策略梯度方向移动，这时梯度更新方向无法正确识别错误动作3。策略梯度算法需要更多的训练样本来消除这类动作造成的影响。

###### 参考资料

[Policy Gradient Algorithms]([Policy Gradient Algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#actor-critic))
