# RL$^2$: FAST REINFORCEMENT LEARNING VIA SLOW REINFORCEMENT LEARNING

# 核心问题
1. 研究问题：如何对reinforcement learning进行learning，也就是meta-learning
2. 假设条件：知道想要解决的问题的模型是什么样（能够构造一大组（分布）MDP）
3. 主要想法：如果在一大组MDP中学习到的agent，在面对未知（新的）MDP时，能有不错的效果，说明agent已经学到这类MDP的性质，也就是prior
4. 解决方案：用RNN来学习，用一大组MDP训练出RNN的权重（视为meta-learning），然后面对新的MDP时，用不断产生的input来调整hidden state，将不断变化的hidden state是为当前MDP的reinforcement learning

# 思路
通常我们都是针对具体类型的问题设计相应的算法，因为是针对具体类型的设计，所以这样的算法必然性能等方面会比较好，但是也因为是对于具体类型设计的，所以必然会有更多的局限性，对于别的类型可能并不适用，所以我们会想：能不能有万能算法能够针对不同类型的问题，学习出相应的类型，然后自己根据问题类型，来学习设计出算法？

deep learning已经具有一定提取特征的能力了，所以懒惰的我们肯定会想如果agent能够根据问题自己调整出网络结构那该多好，这有点跑题了，但是也就是这篇文章，或者meta-learning所希望有的能力：learning to learn。

不同类型的问题和不同类型的算法有不同的学习范式。对于Reinforcement Learning当然最重要的一点是：MDP，因为RL的目的是针对特定的MDP学习出最优或者不错的policy。

那么我们希望能够让agent学习到RL的能力，那其实就是希望能够学习到在不同MDP中寻找最优policy或者不错policy的能力。这其实有点大，或者说有点难，因为MDP的各个部分其实有很多的伸缩性，所以对training等方面是有很大的挑战的。

所以这篇文章其实做了一个假设：我们事先知道要解决是什么问题，根据这个问题的类型，构造了一堆MDP，然后在这堆MDP中学出MDP中特性，然后应用在这个问题上，说明我的agent已经学习到了这类MDP的性质。

# 具体做法
### PRELIMINARIES
discrete-time finite-horizon discounted Markov decision process (MDP) ，$(S, A, P, r, \rho_0, \gamma, T)$:
* $S$： state set
* $A$：action set
* $P$：$S \times A \times S \rightarrow R_+$ transition probability distribution
* $r$：$S \times A \rightarrow [-R_max, R_max]$bounded reward function
* $\rho_0$：$S \rightarrow R_+$: initial state distribution
* $\gamma$：$gamma \in [0, 1]$ discount factor

### RL$^2$
如果我们使用RNN来构造agent，agent的输入为：以前的rewards，actions，termination flags和normal received observations。同时RNN的hidden state并不在每次episode开始后重置，而是保留，然后使用标准的RL算法来training这个agent，那么这个agent应该能capacity to perform learning in its own hidden activations。然后这个agent在deployed时，面对未知的MDP应该能够根据当前的信息来调整hidden state，这也就是学习到了RL的能力，所以也就是这篇文章叫做RL$^2$的原因。

![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/RL%24%5E2%24%3A%20FAST%20REINFORCEMENT%20LEARNING%20VIA%20SLOW%20REINFORCEMENT%20LEARNING/media/1.png)

首先定义一个MDPs的知识，也就是在一大堆MDP中每个MDP被抽样的概率: $\rho_M: M \rightarrow R_+$，然后每次我们通过这个$M$来抽取MDP，抽取后将这个MDP固定$n$个episode，比如上面的图就是固定了2个episode，也就是$n=2$。然后再继续抽取新的MDP，这样不断的学。

这里的细节时：agent会使用上一时刻的reward $r_t$, 上一时刻的action $a_t$，上一时刻的termination flag $d_t$（从上图可知，我们时是固定了一个MDP n episode，所以需要明显地知道结束）和当前的state $s_{t+1}$做为agent的输入的。

另外一个重点是：我们是最大化这n个episode的expected total discounted reward，这其实等价于minimizing the cumulative pseudo-regret。

同时因为我们每次都是抽去出的MDP，所以agent并不知道面对的是哪一个MDP，所以agent应该要能够利用历史上的input推测出这个MDP的信息，然后调整policy，也就是hidden state。

### MULTI-ARMED BANDITS
就是经典的多臂赌博机，这里存在很多个臂，然后从这些臂里面抽取出一些臂做为一个赌博机来学习，每个臂被抽出的概率为$p_i$，所以是可以抽取多个赌博机，这就是上面说的MDP的set，我们的目的是：maximize the total reward obtained over a fixed number of time steps

这是个单状态的问题，但是也要平衡探索与利用，因为研究的比较多，同时有rich theory，可以与一些有理论保证，渐进线形最优的policy做对比，If the learning is successful, the resulting policy should be able to perform competitively with the theoretically optimal algorithms.(超参数在后面有)

这里与几个policy做了对比：
1. random
2. gittins index
3. UCB1
4. Thompson sampling(TS)
5. $\epsilon-greedy$
6. $greedy$

同时，我们把所有的true distribution提供给上面需要的算法（RL$^2$除外）

![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/RL%24%5E2%24%3A%20FAST%20REINFORCEMENT%20LEARNING%20VIA%20SLOW%20REINFORCEMENT%20LEARNING/media/2.png)

![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/RL%24%5E2%24%3A%20FAST%20REINFORCEMENT%20LEARNING%20VIA%20SLOW%20REINFORCEMENT%20LEARNING/media/3.png)

还是不错的，但是因为已有的算法mostly designed to minimize asymptotic regret (rather than finite horizon regret), hence there tends to be a little bit of room to outperform them in the finite horizon settings.

另外发现说在$n=500, k=50$和index有一些差距，为了探索是不是学出来的RL不够好，就使用相同网络结构，然后用index来生成数据对网络做SL，发现能学的和index差不多，说明RL学的还是有提升空间的。

### TABULAR MDPS
多臂赌博机是一个单状态的，但是rl是针对sequential decision making的，所以这里就采用随机生成tabular MDP来做测试,这里我们限制state空间为10，action空间为5，rewards follow a Gaussian distribution with unit variance, and the mean parameters are sampled independently from Normal(1, 1) ，transitions are sampled from a flat Dirichlet distribution，然后episode的horizon为T=10

然后与下面比较：
1. random
2. PSRL
3. BEB
4. UCRL2
5. $\epsilon-greedy$
6. $greedy$ 

![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/RL%24%5E2%24%3A%20FAST%20REINFORCEMENT%20LEARNING%20VIA%20SLOW%20REINFORCEMENT%20LEARNING/media/4.png)

![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/RL%24%5E2%24%3A%20FAST%20REINFORCEMENT%20LEARNING%20VIA%20SLOW%20REINFORCEMENT%20LEARNING/media/5.png)

发现还是有一定效果的

### VISUAL NAVIGATION
每个MDP是随机产生的maze，然后目标是也是随机的，但在一个mdp的不同episode下，maze与终点时固定的。reward和cost设置为：找到目标reward+1，如果碰墙，cost掉-0.001，每个时间step，cost掉-0.04。


![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/RL%24%5E2%24%3A%20FAST%20REINFORCEMENT%20LEARNING%20VIA%20SLOW%20REINFORCEMENT%20LEARNING/media/6.png)

然后这里现在5x5的世界中做training，n=2，horizon=250。然后maze是从1000个configuration中产生的。

在test时做了1.在9x9与5x5里面看看效果怎么样，2.将agent运行5个episode看看怎么样
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/RL%24%5E2%24%3A%20FAST%20REINFORCEMENT%20LEARNING%20VIA%20SLOW%20REINFORCEMENT%20LEARNING/media/7.png)
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/RL%24%5E2%24%3A%20FAST%20REINFORCEMENT%20LEARNING%20VIA%20SLOW%20REINFORCEMENT%20LEARNING/media/8.png)


效果还行

### POLICY REPRESENTATION
输入$(s, a, r, d)$，通过函数$\phi(s, a, r, d)$做embedded后做为RNN的输入，RNN的cell采用GRU，然后在接一层全连接，再使用softmax做为激活函数，输入为每个action的概率。另外这里说参数在每个episode开始的时候重置一部分的hidden state，这样的目的其实是说开始和结束必然存在一些不一样，希望学到这部分不一样，但是实际实验并没有效果。

采用off-the-shelf RL algorithm：rllab and TabulaRL，使用first-order implementation of Trust Region Policy Optimization (TRPO)，同时To reduce variance in the stochastic gradient estimation, we use a baseline which is also represented as an RNN using GRUs as building blocks. We optionally apply Generalized Advantage Estimation (GAE)

* hidden activation: relu
* all weights matrices use weight normalization without data-dependent initialization 
* hidden to hidden weight: orthogonal initializaion
* other weight: Xavier initialization
* bias: 0
* the policy and the baseline uses separate neural networks with the same architecture until the final layer, where the number of outputs differ.

#### MULTI-ARMED BANDITS
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/RL%24%5E2%24%3A%20FAST%20REINFORCEMENT%20LEARNING%20VIA%20SLOW%20REINFORCEMENT%20LEARNING/media/9.png)
#### TABULAR MDPS
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/RL%24%5E2%24%3A%20FAST%20REINFORCEMENT%20LEARNING%20VIA%20SLOW%20REINFORCEMENT%20LEARNING/media/10.png)
#### VISUAL NAVIGATION
1. 40 x 30 RGB image, range [-1, 1]
2. 2层Conv：16个filter， size 5 x 5， stride 2
3. 将action embedded到256-dimensional vector然后与2的输出flattened后拼接
4. 256 hidden dense

![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/RL%24%5E2%24%3A%20FAST%20REINFORCEMENT%20LEARNING%20VIA%20SLOW%20REINFORCEMENT%20LEARNING/media/11.png)

