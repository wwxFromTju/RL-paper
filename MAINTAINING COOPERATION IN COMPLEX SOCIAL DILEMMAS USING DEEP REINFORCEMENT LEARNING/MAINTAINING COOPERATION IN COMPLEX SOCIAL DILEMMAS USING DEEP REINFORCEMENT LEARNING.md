# MAINTAINING COOPERATION IN COMPLEX SOCIAL DILEMMAS USING DEEP REINFORCEMENT LEARNING

# 核心问题
1. 研究问题：在complex social dilemmas中如何设计出既能考虑social welfare，又能确保自己payoff的agent
2. 假设条件：能够观察到对手的action，能够事先知道环境
3. 主要想法：如果想要最大化social welfare，那么就是在对手能够合作的时候与对手合作。要确保自己的payoff，就是在对手背叛的时候，自己也背叛，确保自己不会被对手利用
4. 解决方案：1.事先训练出fully cooperative policy与safe policy（defection）的策略，同时获得双方都合作／竞争时的Q与V，2.然后通过对比对手当前action的Q（合作）的值与对手当前action的Q（合作）的值，如果当前action的Q值更高，就说明对手采用背叛的action 3. 累积对手背叛的程度，在一定阈值后，采用K次背叛的策略（K的次数类似惩罚，采用V值进行计算），然后再切换成合作的策略，再次循环。

# background
关于social dilemma，简单地说就是agent们相互合作能够得到比较高的payoff，同时total payoff是最高的，但是不论在什么情况下，采用selfish的action能够比采用合作的action获得更高的payoff，所以agent为了自己的payoff有动机采用selfish的action。但是双方都采用selfish的action时，payoff反而比都合作时候的payoff低，同时total payoff也比较低。

通常对于social dilemma的研究环境，我们都采用是repeat game，就是固定一个矩阵的game，然后不停地玩这个game。在Deepmind的“Multi-agent Reinforcement Learning in Sequential Social Dilemmas”中，将social dilemma扩展到了sequential下，因为环境更复杂了也带来更多的挑战。比如defection和cooperative是体现在一些列actions中，我们很难通过actions来判断是合作还是竞争，同时传统的ISD直接的方法不能直接扩展到SSD下面。

那么我们自然会想到使用DRL。在复杂的mas环境中，我们通常使用DRL+self-play来训练agent，用简单地话说，self-play就是不断重复模拟game，在模拟中控制所有的agent，并不断地improving这些agent的策略，并最终获得训练好的策略，在面对新的，未知的对手时，采用训练好的policy来应对。

# 思路
但是在SSD中，我们并不能简单地使用DRL+self-play来训练agent，主要的原因是一个一直采用cooperative policy的agent容易被对手利用，一个一直采用defection policy的agent最终只能与一个理智的对手达成social dilemma，所以我们应该设计出一种算法，让agent能够根据对手的行为来调整采用cooperative还是defection。

一个经典的做法叫做Tit For Tat（TFT），TFT就是在第一局采用合作，然后在以后的局面中采用对手上一局采用的action。说来简单，但是TFT是一个非常强大的算法，能够与能合作的对手合作，避免被对手利用，同时一旦对手能够选择合作，就会选择合作，并有希望一直保持合作。

从描述中，我们就能很直观地发现TFT并不能直接用在SSD的情况下，主要的原因是环境已经从一个矩阵，变成一个需要做序列决策的环境了。虽然TFT不能直接用，但是我们可以利用TFT的思想来构造一个与DRL相互结合的算法，也就是这里的算法；APPROXIMATE MARKOV TFT（amTFT）。

TFT能够直接用上一局对手的action来选择自己这句的action的主要原因就是，在传统的矩阵形式的game中，action，reward，defection，cooperative这几个其实可以理解为等价的，一者确定之后，其他就确定了，比如我选择defection的action，那么它的reward信息就大致确定了（因为还与对手的action相关），所以在这里我不严谨地说：选择action，本质上也就是在利用reward的信息。

那么就很直观，在矩阵下面我利用reward的信息，在一个序列的决策中，我们就可以利用Q或者V的信息啊，amTFT就是利用Q与V的信息的。

# 具体做法
首先，我们采用selfish的方式，每个agent都是最大化自己的reward的方式，来训练自己的policy，我们可以得到agent的策略$\pi_i^D$, 相应的Q function approximations $Q^i_{DD}(s, a_1, a_2)$。然后我们在采用fully cooperation的方式，agent目标是最大化total payoff的方式来训练策略，同样我们可以得到agent的策略$\pi_i^C$, 相应的Q function approximations $Q^i_{CC}(s, a_1, a_2)$。

那么如果我们要衡量在当前state $s$下，我们合作时(假设我们是$agent_1$，对手是$agent_2$)，对手当前action是否是合作的？我们可以使用双方都合作的时的$Q^i_{CC}(s, a_1, a_2)$, 假设对手采用合作策略$\pi_2^C$时，和当前采用action的Q的差值，如果当前action的Q值，则说明对手采用了竞争的action，即为下式：
$$
d = Q^2_{CC}(s, \pi_1^C(s), a_2) - Q^2_{CC}(s, \pi_1^C(s), \pi_2^C(s))
$$
当d大于0时，我们觉得对手是采用竞争的策略，那么我们可以变化自己的策略为竞争的策略。

在这里amTFT的实际做法并不是像TFT那样，不停地按照对手的action来调整自己的action，而是变化为defect之后，保持defect k step，然后调整为cooperation，再次观察对手的合作程度。

这里就带来两个问题，一个问题是：一次d其实并不准确，容易有很大的误差。另外的问题是：k step的k该如何决定？

第一个问题其实比较好解决，我们可以不停的累积d，直到d的累积和超过一个阈值，我们认为对手是defect，然后再变换为defect的策略。这样通过累积的方法，更加容忍d计算上可能的问题，但是实际上也带来了一定的延迟性。

第二个问题其实也蛮重要的，因为如果采用固定的k，容易被对手考虑全局，在某个s下来利用这个性质。所以这里采用类似惩罚的思想，使用第一个问题中累积的d，来计算我应该惩罚对手多少局，它才会把这几次背叛获得更多的payoff损失掉：
$$
V_2(s′, \pi_1^{D_kC}, \pi_2^{D_kC}) − V (s′, \pi_1^C, \pi_2^C)>\alpha d.
$$
其中$\alpha$为超参数，大于1, $\pi_i^{D_kC}$代表采用D k step后切换成C。所以这里其实是定义了一个下界。

整体的算法如下所示：
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/MAINTAINING%20COOPERATION%20IN%20COMPLEX%20SOCIAL%20DILEMMAS%20USING%20DEEP%20REINFORCEMENT%20LEARNING/media/1.png)

# 实验

![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/MAINTAINING%20COOPERATION%20IN%20COMPLEX%20SOCIAL%20DILEMMAS%20USING%20DEEP%20REINFORCEMENT%20LEARNING/media/2.png)

在这里做了2个实验环境：
coin game：在一个5x5的格子世界中，有两个不同颜色的agent，也有两个不同颜色的coin，agent不论收集到什么颜色的coin，agent都会得到+1的reward，但是如果agent收集到了另外颜色的coin，对应颜色的agent会得到-2的reward。情况下，selfish的agent就是不论什么颜色都收集，合作的agent就是只收集自己的颜色的coin。

Pong Player’s Dilemma (PPD)：就是将Pong扩张成SSD的情况，赢球的agent获得+1的reward，输球的agent获得-2的reward。所以selfish就是努力自己得分，合作就是双方尽力不得分，在中间传球。


在这两个环境中，amTFT达到了我们的目的：与合作的合作，与竞争的竞争，相互直接合作。


![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/MAINTAINING%20COOPERATION%20IN%20COMPLEX%20SOCIAL%20DILEMMAS%20USING%20DEEP%20REINFORCEMENT%20LEARNING/media/3.png)

然后我们研究如果固定一个agent的策略，另外一个agent用selfish的角度，从头开始学会怎么样（结果如上图）。

这里采用的固定的agent的策略为：合作，竞争，amTFT。面对合作／竞争的agent时，selfish学到利用。面对amTFT，selfish最终学到合作。

