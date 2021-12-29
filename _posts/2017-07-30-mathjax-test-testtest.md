---
layout: post
title: MathJax Test
date: 2017-07-30
categories: test
tags: mathjax 
---

mathjax in markdown :)
# 随机策略梯度理论

### 1、策略梯度

**强化学习的目标**是找到一个最优策略，当agent遵循该策略与环境进行交互时收获尽可能多的回报。假设策略由参数 $\theta$ 表示，记为：$\pi(a\|s;\theta)$，那么目标函数 $J$ 可以表示为:

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
