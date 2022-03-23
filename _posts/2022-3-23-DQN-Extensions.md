---
layout: "post"
title: "DQN Extensions"
date: 2022-3-23
categories: "RL"
tags: "DQN" 
---


### **DQN算法**
&emsp; &ensp; [DQN]算法属于Q-learning中的一种。为了解决Tabular Q-learning类型算法难扩展到大规模环境状态的问题，DQN使用神经网络来拟合Q值函数。与一般的监督学习一样，利用神经网络拟合Q值函数需要有一个针对特征输入的期望输出信号。DQN通过下式构造期望输出：

$$\hat Q(s_t, a_t \| \theta_t) = r_t + \gamma \, max_{a}Q(s_{t+1}, a\| \theta_t)  \quad \quad (1) $$

其中r为单步回报， $\gamma$ 为折扣率。

&emsp; &ensp; 为了使得DQN算法在训练时更加稳定， 通常会采用双Q网络的结构。target-Q网络用于计算 $(s, a)$ 下的期望Q值，Q网络用于预测 $(s, a)$ 的Q值。

$$\hat Q(s_t, a_t \| \theta_t) = r_t + \gamma \, max_a Q(s_{t+1}, a \| \theta_t^-) \quad (2) $$

&emsp; &ensp; target-Q可每步根据Q网络参数，采用动量的方式更新。也可以在Q网络更新m步之后，再将Q网络的参数赋予target-Q网络。算法流程如下：

>*1、agent通过 $\epsilon$-greedy 与环境交互，得到交互数据 $(s, a, r, s')$ ;*\
>*2、利用target-Q网络根据式（1）计算 $(s,a)$ 状态动作对值函数的期望值 $Q^r(s, a)$ ;* \
>*3、利用 $Q$ 网络预测状态动作对 $(s, a)$ 的值函数 $Q(s, a)$;* \
>*4、计算预测值与期望值之间的MSE误差，$L(\theta) = (Q(s, a) - Q^r(s, a))^2$;* \
>*5、利用SGD更新 $Q$ 网络的参数；* \
>*6、在更新 $n$ 次 $Q$ 网络之后将target-Q网络与Q网络参数同步。可采用动量的方式。*

### **Double DQN（DDQN）**

&emsp;&ensp; [DDQN] 的作者认为DQN会产生overestimation现象。原因是动作的选择及期望Q值得计算均使用了相同的网络（均来自Target-Q网络）。通常而言，如果对所有的(s, a)的值函数都产生过高的估计，这不会对算法产生任何不良的影响。但如果算法不是对所有(s, a)的值都产生过高估计，而是集中在我们不关心的(s, a)上，那么算法便会找到一个次优的策略。

&emsp; &ensp; DDQN建议动作选择和Q值计算应分别使用不同的网络计算：
$$ \hat Q(s_t, a_t) = r_t + \gamma \, Q(s_{t+1}, argmax_a Q'(s_{t+1}, a \| \theta ^-_t) \| \theta _t) \quad \quad (3)$$

&emsp; &ensp; 除了期望Q值使用式（3）代替式（2）之外，Double DQN的训练过程与DQN算法一致。

### **Noisy networks(ICLR 2018)**
&emsp; &ensp; 在强化学习领域中，探索（exploration）和利用（exploitation）是一对矛盾。我们期望agent能学习到最优策略，这需要agent具备足够强的exploration能力。另一方面，我们希望agent能更好的利用学习到的先验知识，以便更快的收敛。在确定性环境中我们常采用 $\epsilon$-greedy或者加入entropy损失来提高agent的exploration。***在强化学习中，如何平衡exploration和expoitation是重要问题。这篇博文总结了目前常用到的[exploration]方法。*** 然而，[NoiseNet]的作者认为，这些方法只能在局部小范围内进行动作搜索，且仅限于部分线性函数，很难应用到神经网络中。

&emsp; &ensp; 对于一般的神经网络：

$$y^l=wx + b \quad \quad (4)$$

$w$ 为网络参数，$b$ 为偏置。设 $y=f_{\theta}(x)$ 是参数为 $\theta$ 的NoiseNet层。与一般的神经网络不同，它的参数由两部分组成 $\theta=\mu + \Sigma \odot \epsilon$, 可学习的参数为： $\xi=(\mu, \Sigma)$，$\epsilon$是一个均值为0的噪声向量。

$$ y=(\mu^{w} + \sigma^{w} \odot \epsilon^{w})x + \mu^{b} + \sigma^{b} \odot \epsilon^{b}  \quad \quad (5)$$

&emsp; &ensp; NoiseNet的作者提出了两种添加噪声的方式，一种是在层的参数中添加(Independent Gaussian Noise)；一种是在层的输入输出结果中添加（Factorised Gaussian Noise）。Factorised Gaussian Noise比Independent Gaussian Noise计算量小。对于p维输入，q维输出的层，Independent Gaussian Noise需要 $(q \times p + p)$,Factorised Gaussian Noise只需要 $p + q$ 。

![NoiseNet](https://cdn.jsdelivr.net/gh/wtyhainan/blog-img@main/DQN-Extensions/NoiseNet3.jpg)

### **Prioritized replay buffer(ICLR 2016)**
&emsp; &ensp; Replay Buffer是DQN算法的一个重要组成部分。通过将样本存储到一个长队列中，随机从队列中采样出一个batch用于训练Q网络。Replay Buffer被提出的初衷是降低episode间的样本相关性，使得训练样本之间尽可能满足IID条件。

&emsp; &ensp; 使用一个足够大的Replay Buffer来保存训练样本，然后随机地从Replay Buffer中抽取batch样本用于训练。这是ReplayBuffer的一般做法。[Prioritized replay buffer]的作者认为，在一个大的Buffer中采用均匀分布抽取batch样本用于训练的做法使得样本使用率过低，在训练的过程中应该尽可能多地关注那些重要的样本。

&emsp; &ensp; Prioritized replay buffer的核心使如何衡量样本的重要性程度。作者建议使用TD-error来衡量样本重要性，还给出了两种Prioritized replay buffer的策略。一种是Greedy TD-error prioritization；一种是Stochastic Prioritization。

> **Greedy TD-error prioritization** \
*1、计算状态s的TD-error;* \
*2、选择TD-error大的样本，用于训练；* \
*3、使用Q-learning算法更新训练样本的Q值，样本权重与TD-error误差相关，样本的TD-error误差越大，该样本的权重越大；* \
*4、对于那些新加入Buffer的样本，为了保证它们至少在训练中出现一次，它们的TD-error被设置为当前buffer中的最大值；*

&emsp; &ensp; 作者认为Greedy TD-error prioritization具有三个缺点，第一，TD-error只对训练样本进行更新，这就使得如果一个样本在开始时具有较小的TD-error, 那么该样本会在一段很长的时间不会被用于训练；第二，对噪声特别敏感（例如reward是随机时）。第三，集中在experience的一部分具有较高TD-error的样本子集。\
&emsp; &ensp; 为此，作者建议使用Stochastic Smapling Method从buffer选择训练样本。使用Stichastic sampling方法要保证样本被选中的概率与TD-error相关的同时是单调的。每一个样本被选择的概率为：

$$P(i) = \frac{P_i^{\alpha}}{\sum_{i}P_i^{\alpha}} \quad \quad \qquad (6)$$

有两种方式计算 $P_i^{\alpha}$ 。一种是： $ P_i^{\alpha}= \mid \delta_i \mid + \epsilon $ ，$\delta_i$ 为TD-error， $\epsilon$是一个很小的正数，用于保证每一个样本即使没有TD-error也有一定的概率被抽样到; 另一种是： $P_i = \frac{1}{rank(i)}$， $rank(i)$ 为第i个样本根据 $\delta_i$ 的排序（从大到小）。

> **Stochastic sampleing method** 
![Prioritized Replay Buffer](https://cdn.jsdelivr.net/gh/wtyhainan/blog-img@main/DQN-Extensions/Prioritized-Replay-Buffer.jpg)

&emsp; &ensp; 有一点需要注意的是 $w_i$ 。假设我们开始使用uniform sampling方式计算Q值是无偏的，现在我们使用 Prioritization sampling的方式来计算Q值，由于现在的采样方式与先前的uniform sampling不一样，这就使得我们在计算Q值时引入了误差，为了减少这个误差，使用重要性采样定理修正样本权重。

$$w_i = (\frac{1}{N} * \frac{1}{P(i)})^{\beta} /max_i(w_i) \quad \quad(7)$$

>***思考：使用Prioritized replay buffer技术后，DQN能获得state of the art的结果。这说明Prioritized replay buffer不光是使得样本使用率变高，还能提高DQN的性能。为什么呢？因为迫使网络将更多的注意力放置到了难样本的训练上？*** 

### **Dueling DQN(ICLR 2015)**

&emsp; &ensp; 与传统DQN算法直接预测状态动作值函数 $Q(s,a)$ 不同，[Dueling DQN]将传统DQN输出的 $Q(s,a)$ 分解为 $V(s)$ 和动作优势函数 $A(s, a)$ 两部分。通过这种方式，Dueling DQN能评估状态$s$的价值。这对于那些在某些state下采取action都不会对环境产生任何影响的场景特别有用。这使得Dueling DQN能更高效的发现state与action之间的关系。因此，Dueling DQN网络的训练比一般DQN网络要快，特别是动作空间很大的环境下。

![Dueling DQN](https://cdn.jsdelivr.net/gh/wtyhainan/blog-img@main/DQN-Extensions/Dueling-DQN.jpg)

&emsp; &ensp; 如上图左一所示，在该状态s下采取任何动作都无关紧要，因为agent周围没有任何障碍物，传统DQN会迫使agent将注意力放到前方远处的障碍物。对于左三，在该状态下，agent左侧及前方最近处均有障碍物出现，传统DQN还是将注意力放置到前方远处的障碍物上。而Dueling DQN则更加关注agent前方的障碍物。

![Dueling-Net](https://cdn.jsdelivr.net/gh/wtyhainan/blog-img@main/DQN-Extensions/Dueling-DQN1.jpg)

&emsp; &ensp; Dueling DQN网络的输出由(7)式计算。

$$ Q(s, a;\theta, \alpha, \beta) = \hat V(s;\theta,\beta) + [\hat A(s,a; \theta, \alpha)  - \frac{1}{\mid A \mid} \sum_{a'} \hat A(s, a';\theta,\alpha)] \quad \quad \quad (8) $$ 

&emsp; &ensp; Dueling DQN在输出的时候，将优势函数 $\hat A(s, a; \theta,\alpha)$ 减去了其平均值 $\frac{1}{\mid A \mid} \sum_{a'} \hat A(s, a'; \theta, \alpha)$ 。这么做的原因是使 $ \hat A(s, a;\theta, \alpha)$ 合法化。设想，状态 $s$ 的值函数为 $V(s)$, 且 $V(s) > 0$，它是 $Q(s, a)$ 的加权和：

$$ V(s) = \sum_{a} \pi(a\|s)Q(s,a) \quad \quad \quad (9)$$ 

优势函数定义为：

$$A(s, a;) = Q(s, a) - V(s) \quad \quad (10)$$

如果此时 $\hat A(s, a;\theta, \alpha)$ 均大于零,显然是不满足优势函数的条件。

> ***思考：Dueling DQN是否缩小了 $ S \times A $ 空间？使得算法能更容易找到相关性较强的状态-动作对。***

### **Categorical DQN(2017)**
&emsp; &ensp; [Categorical DQN]的作者认为一般的强化学习算法都将值函数 $V(s)$ (或 $Q(s, a)$ )看成了某个概率分布的期望，作者认为在某些情况下，单纯的使用期望的方式不会使得智能体学习到更加符合实际情况的策略。一般的贝尔曼方程可写为：

$$ Q(s, a) = E(R(s, a)) + \gamma E(s', a') \quad \quad (11) $$

概率分布视角下的贝尔曼方程为：

$$ Z(s, a) = R(s, a) + \gamma Z(s', a') \quad \quad (12) $$

与式（10）不同，式（11）中的变量均为概率分布。$Z(s, a)$ 称为值分布函数。 \
&emsp; &ensp; 将状态值表示成概率分布的形式有几点好处 ***(以下的解释大多来自原文，也不知道翻译的准不准确)*** ：\
&emsp; &emsp; &emsp; 1) Reduced chattering; 贝尔曼最优算子具有不稳定性，特别是将函数近似与贝尔曼最优方程相结合式，这种不稳定性会使得策略不收敛。这被Gordon[6]称为chattering现象 ***(不明所以，暂且放着吧！)***。Categorical DQN的作者认为将值函数表示成概率分布的形式，能够缓解这种现象。\
&emsp; &emsp; &emsp; 2) State aliasing; 即使在一些确定性环境中，State aliasing(状态混叠)也可能导致一些随即现象。在PONG这个游戏中，由于State aliasing现象的出现，agent无法准确预测出回报值 ***（因为混叠现象的存在，使得智能体没有一个明确的学习目标？）***。使用概率分布式的贝尔曼方程，可以给agent提供更加稳定的学习目标。\
&emsp; &emsp; &emsp; 3) A richer set of perdictions; 通过将值函数表示成概率分布的形式，agent可以从更加丰富合理的回报中学习。***（总是感觉怪怪的，怎么说才合适呢？）*** \
&emsp; &emsp; &emsp; 4) Framework for inductive bias; 通过使用概率分布式的贝尔曼方程，它允许我们使用更一般的解决问题的框架，在这样的框架中，我们可以对问题加入一些先验知识。\
&emsp; &emsp; &emsp; 5) Well-behaved optimization; \
&emsp; &ensp; 当我们将贝尔曼方程表示成概率分布形式，首先要考虑的问题是：1）如何确定目标状态概率分布；2）如何根据目标状态概率分布更新当前状态概率分布。\

&emsp; &ensp;  Categorical DQN利用最优概率贝尔曼方程计算目标函数：

$$ a^*_{t+1} = argmax_a(\sum_{bar}(CateDQN(s_{t+1}))) \quad \quad (13) $$ 

$$ Z(s_{t+1}, a_{t+1})=CateDQN(s_{t+1}, a^*_{t+1}) \quad \quad (14) $$ 

$$ \hat Z(s_t, a_t)=R_t + \gamma Z(s_{t+1}, a_{t+1}) \quad \quad(15) $$

&emsp; &ensp; 得到$(s_t, a_t)$的目标分布后，计算KL-loss利用SGD更新网路参数。

&emsp; &ensp; 下图为Categorical DQN算法的大致过程。

![categorical DQN](https://cdn.jsdelivr.net/gh/wtyhainan/blog-img@main/DQN-Extensions/Categorical-DQN.jpg)

>Categorical Algorithm \
input A transition $ (x_t, a_t, r_t, x_{t+1}), \gamma_t \in [0, 1] $ \
&emsp; 1、$ Q(x_{t+1}, a)=\sum_i z_ip_i(x_{t+1}, a) $ &emsp; &emsp; &emsp; &emsp; ***计算每个动作的Q值*** \
&emsp; 2、$ a* \leftarrow argmax_a Q(x_{t+1}, a) $ &emsp; &emsp; &emsp; &emsp; &emsp; ***找到最优动作***\
&emsp; 3、$ m_i=0, i \in 0, ... , N-1 $  &emsp; &emsp; &emsp; &emsp; &emsp; &ensp; &ensp; ***初始化一个全为零的直方图***\
&emsp; 4、for ***j*** in range(N-1): &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; ***对应于上图中的 (a)~(c) 步骤*** \
&emsp; &emsp; &emsp; $ \hat \Gamma z_j \leftarrow [r_t + \gamma_t z_j]^{V_{max}} _ {V_{min}} $ &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; ***当前bar刻度在收到回报 $r_t$ 之后的变化*** \
&emsp; &emsp; &emsp; $ b_j \leftarrow \frac{(\hat \Gamma z_j - V_{min})}{\Delta z} $  &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &ensp;&emsp; ***bar刻度对应的索引***\
&emsp; &emsp; &emsp; $ l \leftarrow \lfloor {b_j} \rfloor, u \leftarrow \lceil {b_j} \rceil $ &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; ***离散化bar索引*** \
&emsp; &emsp; &emsp; $ m_l \leftarrow m_l + p_j(x_{t+1}, a^* ) (u-b_j) $ &emsp; ***计算目标概率分布***\
&emsp; &emsp; &emsp; $ m_u \leftarrow m_u + p_j(x_{t+1}, a^* ) (b_j-l) $ \
&emsp; 5、 $ loss=- \sum_i m_i log(p_i(x_t, a_t)) $ &emsp; &emsp; ***计算损失***


#### **参考文献**
[1]:[Noise Networks For Exploration](https://arxiv.org/pdf/1706.10295.pdf) \
[2]:[Prioritized experience replay](https://arxiv.org/pdf/1511.05952.pdf) \
[3]:[Dueling network architectures for deep reinforcement learning](https://arxiv.org/pdf/1511.06581.pdf) \
[4]:[A Distributional Perspective on Reinforcement Learning]() \
[5]:[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf) \
[6]:[Stable function approximation in dynamic programming](https://www.ri.cmu.edu/pub_files/pub1/gordon_geoffrey_1995_2/gordon_geoffrey_1995_2.pdf)




[DDQN]: https://arxiv.org/pdf/1509.06461.pdf
[DQN]: https://arxiv.org/pdf/1312.5602.pdf
[exploration]:https://lilianweng.github.io/posts/2020-06-07-exploration-drl
[NoiseNet]:https://arxiv.org/pdf/1706.10295.pdf
[Prioritized replay buffer]:https://arxiv.org/pdf/1511.05952.pdf
[Dueling DQN]:https://arxiv.org/pdf/1511.06581.pdf
[Categorical DQN]:https://arxiv.org/pdf/1707.06887.pdf



