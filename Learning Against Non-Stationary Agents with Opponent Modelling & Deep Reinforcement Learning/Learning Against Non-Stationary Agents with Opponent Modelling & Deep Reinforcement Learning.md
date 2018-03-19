# Learning Against Non-Stationary Agents with Opponent Modelling & Deep Reinforcement Learning

# 核心问题
1. 研究问题：在复杂的环境下，怎么对于一个会改变policy的agent做出best response？
2. 假设条件：可以观察对手的action；通过action的变化可以（比较清晰）推测对手policy的变化；对手的policy改变为：1. 缓慢改变的 2. 瞬间改变，但是保持一段时间；
3. 主要想法：采用DRL来解决复杂环境的问题，结合opponent model来切换策略
4. 解决方案：事先设定训练n个opponent model，通过opponent model来检测对手的policy类型，然后改变自己的相应的response。每个opponent model对应对手的一种类型的policy，同时使用DDPG来训练这种类型的policy的best response。

我认为其实就是披着DDPG衣服的：连续状态空间的opponent model的做法，即利用uncertainty estimates，multiple policies来根据对手的policy来switch policy。本文的贡献和DRL其实没有太大的关系。


# 动机
现实世界本质上是一个multiagent的环境，在multiagent的环境中，就必然存在的agent之间的相互交互，在相互交互时，很多情况情况下agent需要决定是与另外一个agent合作还是竞争。所以在这种环境中希望最大化自己的收益，那么应该观测别的agent的行为，并来调整自己的policy。

与single agent中常见的环境不会发生的设定不同，因为其他的agent也可能改变的自己的policy，进而影响环境的变化，所以这样的环境可以被形式化称为：decentralised non-stationary learning problem。为了解决这样的环境，一项很重要的能力就是检测另外agent的缓慢／突然行为的变化。

single agent的DRL直接运用在这种环境中效果会非常差，主要的原因就是：single agent解决的环境就是stationary的，所以直接使用过来必然会有一些gap。上面我们知道一种影响stationary的可能原因是：agent的策略变化，所以如果想要利用当前single agent的做法，那么直观上就会考虑：是否能不能将这个non-stationary reduce掉？

所以篇文章提出SAM来显示检测对手的policy的变化，然后根据变化来做调整，reduce这个non-stationary。更具体的做法是：建立opponent model，然后通过uncertainty estimations来检测另外agent的policy的switch。

# 为什么opponent model有效？

通过建立opponent model可以预测对手在当前state下会采取什么样的action，如果对手的policy并不是一个无懈可击的policy，那么我们通过opponent model可以发现他们存在一些suboptimal behaviour，那么就这个针对这个behaviour来做处针对的behaviour，以此来获得更多的reward。

# 在Non-Stationary下，我们希望opponent model具有什么性质？
当前的opponent model已经在一些论文中使用（比如maddpg，LOLA），这些工作结合DL的opponent model的有效性，解决了一些相对复杂环境下learning的问题。但是这些使用并没有考虑到：non-stationarity的环境中该如何利用opponent model。

在non-stationarity的环境中，因为另外的agent的policy可能也会不停的更新，所以一个很重要的特性就是opponent model能够连续的更新。此外，这个oppoent model应该能检测两种类型的对手：
* sudden：突然改变，即切换策略
* gradual：缓慢改变，比如也是不停在学习

比如在一个未知的新的环境开始learning，agent开始时候会缓慢调整自己的policy，在学习过程中能够了解这个环境是什么样的，所以在对这个环境比较熟悉后，可能会突然切换自己的policy。

# Switching Agent Model (SAM)
这里提出switching agent model（SAM)的方法来结合DRL与opponent model。通过approximate bayesian neural network，直接对（s，a）的trajectory来学习，然后通过Monte Carlo dropout来获得predictions的uncertainty。这样就能够robustly地检测出另外的agent是否改变了行为，然后switching opponent model和相应的policy。


简单地讲，SAM就是一系列的policy set，每个分别是对手的策略$\hat{u}$和自己对于这个策略的近似最优的策略$u$，然后通过switchboard来切换策略。

##  Switchboard
switchboard就是跟踪opponent model的效果，根据误差来判断对手是否改变了policy。

如果对手变化的policy，那么这个opponent model就会有相应的误差，通过这个误差来检测policy的变化，然后强化相应的opponent model，调整自己的policy为：变化后策略的近似best response。对于两种不同类型的变化，switch是突然产生一道尖峰，notable spike。gradually就是开始有小小的误差，然后累积。

针对上面的情况，就可以使用规则来判断对手policy的变化了。可以设置阈值$r_{max}$，如果预测误差直接超过阈值，那么我们可以猜测对手改变了policy，接下去就是切换策略。因为是两种类型的变化，所以这里简单地做了一些假设与规则：如果是缓慢变化的，那么我们可以每次都预测对手的action，同时与对手真实的action作对比，通过不同action的误差r来判断对手是否switch了（即这个误差很大），如果没有switch（即误差比较小，但是对手可能是缓慢变化的），就将误差累加起来，并减掉一个定值d。如果变化，那么就切换opponent model，同时将之前的误差清零。
![](https://github.com/wwxFromTju/RL-paper/raw/master/Learning%20Against%20Non-Stationary%20Agents%20with%20Opponent%20Modelling%20&%20Deep%20Reinforcement%20Learning/media/1.png)
其中：
$$
r = r + \frac{a_{t}^{j}-\hat{a}_{t}^{j}}{\eta_{t}} 
$$
$\hat{a}_{t}^{j}$为action prediction，$a_{t}^{j}$为really take action，$\eta_{t}$为associated predictive uncertainty，$r$为running error。

## Response Policies
使用DDPG来训练每个opponent model对应的policy，分开policy的一个好处：避免要训练一个特别robust的policy的难度，同时可以更好地针对每种opponent model做出更好的response。但是因为要训练更多东西，所以速度就会更慢一些。

具体的一点做法是，针对每个opponent model都保存一个对应的replay buffer $D^{k}$，然后使用这个replay buffer训练相应的policy。

然后一个值得注意的地方是：除了state的信息，这里的policy把另外agent的action的预测也当成policy的输入。（这样很像maddpg，reduce掉non-station）。表示为：
$$
a_{t} = u^{k}(s^{i}_{t}|| \hat{a_{t}}) + N_{t}
$$
$N_{t}$为noise

## Opponent Models
![](https://github.com/wwxFromTju/RL-paper/raw/master/Learning%20Against%20Non-Stationary%20Agents%20with%20Opponent%20Modelling%20&%20Deep%20Reinforcement%20Learning/media/2.png)

一个比较有意思的转换是：利用是continuous actions的假设，所以预测对手的action就变成一个回归的问题。那么这个model的uncertainty就可以表示为对于predictions的confidence interval，所以训练时就使用均方误差来作为loss，然后最小即可。

在使用时，使用类似boosting的做法，将一个state输入网络N次，每次hidden unit的dropout的概率都为p，或者k个预测的action，然后求平均值。同时使用sample variance作为model的uncertainty的近似，然后与固定的noise（比如上面说的使用noise来探索）求和，开方。


# 实验
为了能够证明就是将上面对手的类型的考虑来设计相应的实验，说明SAM的确有效：
1. 对手不断的switches between policies，证明SAM的确能够identify，track，adapt
2. agents一起学习，即缓慢的变化。

与传统的从头学习不同，这边主要的假设是对手会改变策略，所以agent的主要目的是identify对手的policy，同时track它的改变。
下面的实验内容：一个是与切换策略的对手，一个时与传统DDPG直接对比

##实验环境

![](https://github.com/wwxFromTju/RL-paper/raw/master/Learning%20Against%20Non-Stationary%20Agents%20with%20Opponent%20Modelling%20&%20Deep%20Reinforcement%20Learning/media/3.png)

一个收集apple的环境：收集apple获得比较小的reward，通过偷别人的水果获得比较大的reward。所以合作的策略是：一直收集水果，不去偷别人。竞争的策略是：去偷别人的reward。从最大化自己的收益的角度：面对竞争的对手时，应该去收集水果，同时避免被偷reward。面对合作的对手时，去偷它的reward来获得比较大的reward


为了加速学习的速度，这里是环境的设置与除了收集水果与偷之外的reward的设置：
1. small negative reward proportional to their distanceto the nearest apple
2. small negative reward proportional to their distance to the opposingagent when it is possible to steal

## Experiment
![](https://github.com/wwxFromTju/RL-paper/raw/master/Learning%20Against%20Non-Stationary%20Agents%20with%20Opponent%20Modelling%20&%20Deep%20Reinforcement%20Learning/media/4.png)
![](https://github.com/wwxFromTju/RL-paper/raw/master/Learning%20Against%20Non-Stationary%20Agents%20with%20Opponent%20Modelling%20&%20Deep%20Reinforcement%20Learning/media/5.png)


因为DDPG只学习一个策略，所以学习速度会比同时学习两个策略的快，所以在开始的时候DDPG的曲线会高一些，但是在后期因为只采用一个策略来面对两种类型的对手，并不是一种最优的选择，而是suboptimal，所以曲线会相对低。

相应的SAM是针对两种类型的agent学习出相应的policy，所以在前期可能会慢一些，但是在后期，因为可以针对对手的policy来调整策略，所以曲线会高一些

为了证明opponent model是否是有效的，单独将training后的检测模型拿出来检测对于switch的效果，发现的确能够有效的检测出相应的policy的变化

之后将DDPG与SAM同时训练，然后比较在训练后，相互play的效果，发现SAM的效果更好，说明能够reduce环境的non-stationary

分析一下通过不确定性是否对的：随着训练的进行，不确定性下降


