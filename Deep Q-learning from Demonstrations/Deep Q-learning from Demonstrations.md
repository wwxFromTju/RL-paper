# Deep Q-learning from Demonstrations

# 核心问题
1. 研究问题：如何加速agent的学习过程，避免前期的cold start
2. 假设条件：可以事先获得一堆的Demonstrations，知道reward function，
3. 主要想法：利用Demonstrations 来per-training NN的weights来缓解cold start
4. 解决方案：利用reward function标记Demonstrations，构造NN的loss function为：RL的loss + Demonstrations的loss， 来初始化NN的Q function，然后在training时候采用同样的loss function来衔接


# 动机
从cold start与data cost的角度来考虑，如果是模拟器的环境，那么DRL可以直接用，因为模拟器的cost与时间成本低，最多就多用几台机器，多开几个并行的环境。但是很多real world的问题，需要learning快，不然前期因为学习时间与data cost导致成本非常高，导致DRL运用存在很大的gap，更直观的理解就是：agent需要很多的step与探索来获得一个对当前交互env的比较全面的认识，但是在真实世界，考虑时间／经济成本，我们希望agent学习的越快越好。所以想在real world的环境中使用DRL就必须解决训练速度的问题，那么cold start就是一个切入点。

考虑到在实际中，我们可以事先获得一些好的数据／策略（比如人类专家，之前运行的算法／规则），从这个角度出来，一种直观的想法利用这部分的信息来per-train NN的weight，然后再与环境做交互来进一步提升效果，应该能够缓解cold start与训练成本的问题。


# 算法
## 算法流程
Deep Q-learning from Demonstrations（DQFD）分为两个阶段：per-training，acting on the system。两个部分采用同样的loss的$J$，
![屏幕快照 2018-06-09 下午12.20.52](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Deep%20Q-learning%20from%20Demonstrations/media/1.png)

### per-training阶段
从Demonstrations抽样来对于NN的Q function做sgd，过程类似于的监督学习。
### acting on the system阶段
1.采用per-training出来的NN，与环境做交互，然后将data存于单独的replay中
2.然后在training时，利用proportional prioritized sampling从Demonstrations与交互生成的data中抽样
3.training NN，这一步类似于DQN，直接做SGD，就是loss function $J$ 做了一定的修改（见后面）
4.在replay满了之后，扔掉旧的交互数据。

# loss $J$的构造
$$
J(Q) = J_{DQ}(Q) + \lambda_1J_n(Q) + \lambda_2J_E(Q) + \lambda_3J_{L2}(Q)
$$


## RL： one-step，n-step
其中$J_{DQ}(Q)$为不同Double DQN产生的one step的loss，double的做的做法可以缓解Max op带来的Q funciton过高估计的问题。
$J_n(Q)$为n-step Q-learning的loss:
$$
J_n(Q) = (r_t + \gamma r_{t+1} + ... + \gamma^{n-1}r_{t+n-1} + max_{a}\gamma^{n}Q(s_{t+n},a) - Q(s_t, a))^2
$$
通过N-step的做法，本身就是相当于将N step中reward快速回传回来，一定程度缓解credit assignment的问题，加速网络的学习，但是我觉的这边因为是off-policy的更新方法，会引入bias，论文中并没有对应的解决／缓解方法，需要思考一下。

## Demonstrations
$$
J_E(Q) = max_{a} [Q(s,a) + l(a_E, a)] - Q(s, a_E)
$$
其中$a_E$为Demonstrations中的action。

使用专家数据来做supervised loss其实有利有弊，因为实际上这部分数据只包含了state，action空间的一小部分。但是如果只用Q-learning来做per-training，由于MAX op的存在，NN的目标可能会选择那些ungrounded variables，然后因为bootstrap的原因，将其传播到其他的State下。

权衡一下，利用SL与RL中的Bellman equation，将专家数据看成一个软的初始化约束，在per-training的时候，约束专家数据中的action要比这个state下其他的action高一个$l$值。这里其实是做了一个loss的权衡:这个$l$值导致的action差别的loss高，还是不同action导致达到下一个状态的$S’$的产生的loss高，如果专家的SL带带来loss高的话，那么以专家的loss为主，如果是RL的loss高的话，那么这个约束就会被放宽，选择RL中的action。


## loss总结
pre-training的阶段目的是对于已有专家数据做imitate，所以为了能够与之后与环境交互部分结合（online RL部分），这边imitate从单纯学习action，转为学出对应的Q function。采用了1-step／n-step Q-learning 来做loss，还有对于专家数据的SL，还有对于NN的weight的L2范数。因为Q-learning部分能够比较好地利用per-training出来的Q function，同时利用专家数据来限制策略的空间，避免spg过程的policy性能的恶化（可能导致螺旋性的恶化）。

# 实验

![屏幕快照 2018-06-09 下午12.46.30](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Deep%20Q-learning%20from%20Demonstrations/media/2.png)
![屏幕快照 2018-06-09 下午12.46.35](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Deep%20Q-learning%20from%20Demonstrations/media/3.png)
![屏幕快照 2018-06-09 下午12.46.40](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Deep%20Q-learning%20from%20Demonstrations/media/4.png)

