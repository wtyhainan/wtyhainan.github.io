---
layout: post
title: Deep Q-Network
date: 2022-1-15
categories: RL
tags: "policy-gradient" 
---


# Deep Q-Network（DQN）

## 1、什么是Deep Q-Network?
&emsp;&emsp;DQN 是 DeepMind 于 2013 年发表在 NIPS 上的一种强化学习方法，该方法首次将深度学习与强化学习结合用于解决大规模（或连续状态）强化学习问题。\
&emsp;&emsp;值迭代算法利用贝尔曼方程来建立动作值函数或状态值函数。值迭代算法从任意初始化值函数开始，采用最优贝尔曼方程迭代求解值函数，经过多次迭代收敛到最优值函数。Q-leanring算法:
\
&emsp;&emsp;1、建立一个空表用于保存 $ Q(s,a) $ 。\
&emsp;&emsp;2、与环境交互。获取 $ (s,a,s',r) $ 数据。在这一步，智能体需要根据状态 $ s $ 确定要执行的动作 $ a $ 。\
&emsp;&emsp;3、更新 $ Q $ 值。 $ Q(s,a) = r + \gamma \, max_{a' \in A}Q(s', a') 。$\
&emsp;&emsp;4、重复以上步骤，直至收敛。
\
上述算法中Q值的更新方式常会导致Q值过度震荡，可采用指数加权的方式更新Q值，使其平滑收敛。\
$$
Q(s,a) = (1-\alpha)Q(s,a) + \alpha(r + \gamma \, max_{a' \in A}Q(s', a'))
$$  

&emsp;&emsp;传统 Q-learning 方法在解决大规模问题时存在以下缺点： \
&emsp;&emsp;1)需要更多的存储空间用于保存 $Q(s,a)$ 值；\
&emsp;&emsp;2)要求环境状态必须是离散的。对于连续状态而言，必须先对其离散化才能使用。\
&emsp;&emsp;为了解决 Q-learning 存在的问题，DQN 使用神经网络来计算 $ Q(s,a\|\theta) $ 值。DQN的输入为 $ s $ ，输出的每一个神经元对应于每一个动作的 $ Q(s,a) $ 值。使用最优贝尔曼方程计算参考 $ Q_{r}(s, a \| \theta) $ 值，然后利用MSE误差训练 $ Q $ 网络。
## 2、DQN原理
&emsp;&emsp;DQN采用神经网络近似 $Q$ 值函数。将t时刻的状态 $ s_t $ 输入给Q网络，Q网络针对agent的每个动作均输出一个 $Q$ 值。为了训练Q网络，我们需要利用贝尔曼方程得到参考Q值，然后利用SGD调整Q-network参数。能否将深度学习技术应用到强化学习问题中的关键有两点：一是能否从即时回报中学习到当前动作与延时回报之间的关系；二是用于训练的数据是否服从独立同分布条件。
<div align="center">

![DQN示例](./figure1.jpg) 图1 Q-network

</div>

### 2.1 最优贝尔曼方程
&emsp;&emsp;监督学习需要采用大量标签数据来训练网络。在强化学习问题中，agent通过策略与环境交互的过程中只收到环境给与的即时奖励，如何使用深度学习技术来使得agent学习到动作与延时回报（延时回报可能发生在成百上千个动作之后）之间的关系呢？参考传统 Q-learning 算法中的第三步，DQN算法也使用最优贝尔曼方程来计算 $t$ 步的最优 $ Q^* $ 值，然后以 $ Q^* $ 最优参考调整 $ t $ 步的 $ Q(s,a \| \theta) $ 网络参数。DQN使用下面的步骤来更新 $Q$ 网络。
> &emsp;&emsp;1、agent基于策略 $\pi(a\|s)$ 与环境交互得到样本数据 $D$；\
> &emsp;&emsp;2、对于第 $i$ 个batch的数据 $ (s_t, a_t, r_t, s_{t+1}) $ 使用贝尔曼方程计算最优 $Q$ 值: $ Q^* = r_t + \gamma\,max_{a_{t+1}}Q(s_{t+1}, a_{t+1}\|\theta_i) $ ;\
> &emsp;&emsp;3、计算当前网络输出的 $ Q(s_t,a_t\|\theta_{t}) $ 与 $ Q^* $ 之间的MSE误差; $ L_t(\theta_i)=E_{s,a \sim \rho( * )}[(Q^* - Q(s_t,a_t\|\theta_i))^2] $ ；\
> &emsp;&emsp;4、使用SGD更新 $ Q(s,a\|\theta_i) $。
### 2.2 experience replay mechanism (经验回放技术)
&emsp;&emsp;当我们使用深度学习来拟合Q值函数时，相当于将其转换为监督学习问题。对于监督学习问题，算法要求其样本服从IID（独立同分布）。对于强化学习问题而言，agent基于一定的策略 $\pi(a\|s)$ 与环境交互，得到episodes。对于不同的 $ \pi(a\|s) $，episode服从的概率分布均不同，因此不同的episodes之间不满足同分布要求。对于同一个episode而言，前后数据之间存在较高的相关性，故其不满足独立性要求。为此，DQN的作者使用一种称之为经验回放的技术来缓解训练数据不满足IID条件的问题。DQN开辟了一块FIFO（先进先出队列）内存空间用于存储agent每一次与环境交互得到的数据，然后使用均匀分布从FIFO队列中采样得到一个batch数据用于训练 $ Q $网络。经验回放技术极大地缓解了数据不满足IID条件问题，是DQN算法成功应用的关键性技术。
### 2.3 $\epsilon$-greedy贪婪策略
&emsp;&emsp;DQN算法使用 $\epsilon$-greedy策略与环境交互。为了使 $Q$ 网络收敛到最优值，agent需要使用不同的动作与环境交互以此来获得更多的环境信息，这就需要agent具备很强的探索能力，随机策略是一个很好的选择。然而，为了使agent更好地利用已知样本信息，更快地收敛，agent需要根据学习到的 $Q$ 值来决定需要执行的动作，这体现了agent的exploitation能力。$\epsilon$-greedy贪婪策略常用来平衡agent的exploration和exploitation能力。在初始阶段，智能体以一个较大的概率 $\epsilon$ 采用随机动作与环境进行交互，以获得尽可能多的环境信息。随着 $Q$ 值网络收敛，$\epsilon$ 逐渐降低，agent更多时候是采用贪婪策略与环境交互。
## 3、算法流程
> 1、设置 $\epsilon$-greedy的初始值及衰减步数;\
> 2、随机初始化Q-network参数；\
> 3、设置Experience Replay Buffer(ERB)大小；\
> 4、设置batch_size大小;\
> 5、for episode in range(episodes): \
> &emsp;&emsp;&emsp;初始化环境 $ s_0 $ ;\
> &emsp;&emsp;&emsp;for t in range(T): \
> &emsp;&emsp;&emsp;&emsp;&emsp;agent根据 $ \epsilon $ -greedy策略与环境交互，得到交互数据 $ (s_t,a_t,r_t,s_{t+1}) $ ，并放入ERB ;\
> &emsp;&emsp;&emsp;&emsp;&emsp;if len(ERB) < batch_size:\
> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;continue ;\
> &emsp;&emsp;&emsp;均匀的从ERB中抽样，得到一个batch_size大小的训练数据，记为 $ \small (s^b_ {t},a^b_ {t},r^b_ {t},s^b_ {t+1}) $；\
> &emsp;&emsp;&emsp;利用贝尔曼方程计算参考Q值， $ \small Q_r(s^b_ {t}, a^b_ {t})=r_t+\gamma \, max_{a^b}Q(s^{b}_ {t+1}, a^{b}\|\theta_t) $ ;\
> &emsp;&emsp;&emsp;利用Q-network计算 $ Q $ 值，$ \small Q(s^b_t, a^b_ {t})=Q(s^b_ {t}, a^b_ {t}\|\theta_t) $ ;\
> &emsp;&emsp;&emsp;计算 $ \small L(\theta_t)=\frac{1}{2}(Q(s^b_ {t},a^b_ {t})-Q_r(s^b_ {t},a^b_ {t}))^2 $;\
> &emsp;&emsp;&emsp;更新 $ \small \theta_{t+1}=\theta_t - \alpha \, \nabla_{\theta_t}L(\theta_t)=\theta_t - \alpha\, \nabla_{\theta_t}Q(s^b_ {t},a^b_ {t})(Q(s^b_ {t},a^b_ {t})-Q_r(s^b_ {t},a^b_ {t})) $ 

&emsp;&emsp;在具体实现时，为了使得DQN训练时更加稳定，通常会才用双Q-network的结构。使用target Q-network来计算参考Q值，并在k次更新Q-network之后，再将target Q-network网络与Q-network网络同步。
## 4、DQN优缺点
&emsp;&emsp;DQN使用神经网络来拟合Q值函数，充分利用了神经网络强大的拟合能力，能较好地适应各种复杂情况；属于off-policy模型，样本利用率高。DQN的缺点同样明显，就是有时候会过高估计参考Q值。
## 5、总结
&emsp;&emsp;DQN属于第一个将深度学习融合进强化学习的模型，开辟了深度学习与强化学习相结合的先例，成功解决了许多连续状态（或大规模状态）下的强化学习问题。应用DQN模型，使得agent在部分Atari游戏上的表现超过了人类。
#### 参考资料
[1] [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)
