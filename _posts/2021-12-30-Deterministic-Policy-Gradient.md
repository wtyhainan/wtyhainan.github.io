---
layout: post
title: deterministic policy gradient
date: 2022-1-1
categories: RL
tags: "policy-gradient"
---



# 确定策略梯度理论

## 1、Deterministic Policy Gradient（DPG）
策略梯度算法广泛应用在连续动作空间的强化学习问题中，策略梯度算法的核心思想是利用一组参数来描述策略，随机梯度策略算法（SPG）将策略表示为动作action的概率密度函数，记为 $ \pi(a\|s;\theta) $ ,agent从 $ \pi(a\|s;\theta) $ 中采样获得在某个状态 $ s $ 下的动作 $ a $ ，执行动作 $ a $ 与环境交互并将获得的样本用于优化参数，使其最大化agent的折扣回报。然而，SPG算法需要从概率密度函数中采样获得，在高维场景下，采样往往是困难的。而且，SPG属于on-policy方法，样本使用率过低。
<br>
确定性策略梯度算法将策略表示为关于状态 $ s $ 的函数，记为 $ a=\mu_{\theta}(s) $。确定性策略函数给出状态 $ s $下agent要执行的动作 $ a $，而不再是一个关于动作 $ a $ 的概率密度函数。 这样做的一个好处是，agent的动作不再是从概率密度中采用得到，避免了高维动作空间难采样问题。


## 2、DPG理论推导

假设随机策略函数表示为：$ \pi(a \| s;\theta) $，agent遵循该策略与环境交互，得到完整episode：
<br>
$$ h_{1:T} = s_1, a_1, r_1, s_2, a_2, r_2,...,s_T,a_T, r_T $$
则对于该次episode的折扣回报 $ G^{\gamma}(\pi) $ 可以表示为：

<br>

$$
\begin{aligned}
G^{\gamma}(\pi) &= p(s_1)\pi(s_1, a_1)r_1 + 
\gamma p(s_1)p^{\pi}(s_1 \rightarrow s_2,1)\pi(s_2,a_2)r_2 + 
\gamma^2 p(s_2)p^{\pi}(s_2 \rightarrow s_3, 1)\pi(s_3, a_3)r_3 + ... \\
&=
p(s_1)\pi(s_1, a_1)r_1 + 
\gamma p(s_1) p^{\pi}(s_1 \rightarrow s_2, 1) \pi(s_2, a_2)r_2 + 
\gamma^{2} p(s_1) p^{\pi}(s_1 \rightarrow s_2, 1)\rho^{\pi}(s_2 \rightarrow s_3, 1)\pi(s_3, a_3)r_3 + ... \\
&= p(s_1)\pi(s_1, a_1)r_1 + \gamma p(s_1)p^{\pi}(s_1 \rightarrow s_2, 1) \pi(s_2, a_2)r_2 + \gamma^{2} p(s_1) p^{\pi}(s_1 \rightarrow s_3, 2) \pi(s_3, a_3)r_3 + ... \\
\end{aligned}
\tag{1}
$$

<br>

令折扣概率 $ \rho^{\pi}(s^{'}) = \int_{s} \sum \limits_{t=0}^{\infty} \gamma^{t} p(s) p^{\pi}(s \rightarrow s^{'}, t) ds$，则目标函数 $ J(\pi) = E(G^{\gamma} \| \pi)$ 可以写为：

<br>

$$
\begin{aligned}
J(\pi) = \int_{s \in S} \rho^{\pi}(s) \int_{a \in A} \pi(a|s ;\theta) \; r(s, a) \; dsda
\end{aligned}
\tag{2}
$$

现假设我们的策略函数不再是action的概率密度函数，而是关于环境的确定性动作，记为 $ \mu_{\theta}(s) $，那么（2）式中关于动作 $ a $ 的积分就可以消去，（2）式可写成：

<br>

$$
\begin{aligned}
J(\mu_{\theta}) &= \int_{s \in S} \rho^{\mu}(s) \; r(s, \mu_{\theta}(s)) \; ds \\\\
 &= E_{s \thicksim \rho^{\mu}}[ r(s, \; \mu_{\theta}(s))]
\end{aligned}
$$

<br>

因此由链式法则，目标函数 $ J(\mu_{\theta}) $ 关于参数 $ \theta $ 的导数可写为：

$$
\begin{aligned}
\nabla_{\theta} J(\mu_{\theta}) &= \nabla_{\theta} \int_{s \in S} \rho^{\mu}(s) \; r(s, \mu_{\theta}(s)) \; ds \\
&= \int_{s \in S}\rho^{\mu}(s) \; \nabla_{\theta}r(s, \mu_{\theta}(s)) \\
&= \int_{s \in S} \rho^{\mu}(s) \; \nabla_{a}r(s, a) \; \nabla_{\theta}\mu_{\theta}(s) |_{a=\mu_{\theta}(s)} \; ds \\
&= E_{s \thicksim \rho^{\mu}}[ \nabla_{a}r(s, a) \; \nabla_{\theta} \mu_{\theta}(s) |_{a=\mu_{\theta}}]
\end{aligned}
\tag{3}
$$

<br>

在原始的DPG文章中，作者表明DPG只是SPG的一种特例，当SPG中概率密度的方差为0时，SPG与DPG等价。在实际应用中，通常用函数 $ \Phi(s,a) $ 来表示（3）式中的 $ r(s,a) $ 。

![test]( \figures\_post\DPG_1.jpg )

### 总结
1、与SPG天生自带很强的exploration能力不同，DPG对环境的exploration能力相对较弱，对于相同state的输入，DPG输出永远是相同的。通常会在DPG输出部分添加噪声来提高agent的exploration能力。
<br>
2、与SPG一样，DPG同样需要 $ \Phi(s,a) $ 函数来评估action的好坏。与SPG不一样的是，DPG 可以方便采用off-policy的方式来训练，而不需要使用重要性采样。这使得DPG在高维度动作空间中更容易完成。
<br>


#### 参考资料
[policy gradient algorithm](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#actor-critic)
<br>
[Deterministic policy gradient](https://hal.inria.fr/file/index/docid/938992/filename/dpg-icml2014.pdf)





