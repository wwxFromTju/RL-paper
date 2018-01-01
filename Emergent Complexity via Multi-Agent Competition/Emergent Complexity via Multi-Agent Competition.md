# Emergent Complexity via Multi-Agent Competition

# 核心问题
1. 研究问题：在MAS环境下，如何稳定，有效地训练一个具有复杂能力的agent
2. 假设条件：普通的multiagent假设，没有中心式的critic，independent training，不用知道对手的action和policy
3. 主要想法：利用self-play的思想，产生合适的对手。同时利用额外的reward来帮助学习basic action
4. 解决方案：设计Exploration Curriculum，前期利用额外的reward来帮助学习基本action，后期将这部分额外的reward衰减掉，让agent能够学习大宋真正希望学习的任务。使用Opponent Sampling，避免对手一旦学的很好，使的自己的agent无法学习，无法收敛。


其实很多学科最终都是要实用，现在的DRL玩玩Atari的游戏还好，但是面对更难的游戏，比如星际二的时候，其实效果并不好，主要的原因是星际二的环境更复杂，我们需要学习出一个更复杂的agent。抽象地说环境的复杂性与学习出skill的复杂性是相关的。越复杂的环境就能学习出越复杂的skill（但是我觉得复杂的话，有时候很难学！所以我们可能需要更多的工程的努力）


不同的single agent的复杂性只与环境本身相关，比如你总不能让一个普通pixel的pong的agent学出正手，反手击球吧。但是在multiagent的环境中，复杂性其实还与对手是相关的，比如围棋，规则很简单，但是你在和对手博弈的过程是十分复杂的。围棋的新手与高手有很大的区别，如果你让你的agent一直与一个新手玩，这个agent可能最多就战胜这个新手，但是实际上不会很厉害。如果你让你的agent一开始就和柯洁下，按照rl的学习方法，开始探索，然后慢慢学习，估计agent一盘都赢不了，最终什么都没有学到。

所以如何在一个环境里面学习到更复杂的策略，如果有效的学习是这篇文章的目的

# Exploration Curriculum
这里假设环境都只在最后通过结果给予agentreward（后面环境设置都是如此），那么必然就带来reward稀疏的问题。一般的稀疏的reward的问题，通过随机探索可以得到一定缓解，但是如果需要一些基本basic的行为，然后再做操作的任务，可能随机探索就不是很好了

比如一个经典的HRL的例子：
我们希望一个机器人移动到门边，然后推开门，只有在推开门才给reward

描述很简单，但是实际上机器人首先需要学习到控制motor，通过控制motor来移动，然后才是导航，推门的动作。这样的流程其实是比较复杂的，如果单纯随机的话，可能机器人只会在原地打滚

所以就会有一个简单的想法，通过额外的reward来帮助agent学习，比如奖励移动，站立。那么agent就容易获得dense的reward，学习到移动，站立。但是这样的在实际训练中其实是不自然的，同时可能会影响，甚至损害agent在原先任务上的学习。比如：如果给予移动比较大的reward，而出现成功开门的几率又比较小，那么agent可能会被诱导成一直在移动的agent

那么直观地，我们肯定想要权衡两者间的平衡，一种做法就是Exploration Curriculum：
在开始的时候给予dense reward来帮助学习，让agent学习到一些basic的动作，比如移动，站立，同时可以帮助提高真正想学习任务reward出现的几率，这个reward被称为exploration reward。

随着训练，把exploration reward减少到0,即不断减小下面的$\alpha_t$，所以只剩下真正任务的reward，那么接着学，优化的就是真正想要学习的任务了：
$$
r_t = \alpha_t s_t + (1-\alpha_t)\mathbf {1}(t == T)R
$$

# Opponent Sampling
正如上面说到的，太难对手，你太难学，太简单的对手，你也学的菜。那么该如何利用对手来学习出一个复杂的task的policy呢？

在实际学习的时候，可能出现一个agent先学的比较好，那么另外一个agent可能完全学不到，或者不收敛，这种情况出现的比较多。也就是上面的：太难的对手出现的几率会多一些。

所以我们会想对手不要那么难就好了，一种想法是：现在他比较难，但是他之前可能比较简单啊！

这里给了一个解决方案：
就是在每个rollout开始的时候，选择random old version的对手来学习。这样的效果能学习得更稳定，策略更鲁棒。

Note that for self-play this means that the policy at any time should be able to defeat random older versions of itself, thus ensuring continual learning. 

# Training Competitive Agents
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Emergent%20Complexity%20via%20Multi-Agent%20Competition/media/1.png)

这篇文章采用distrubuted PPO对agent做优化

一个比较有价值的点是：PG在multiagent的设置下更容易受到方差的影响而不稳定，为了解决这个不稳定的问题，MADDPG用了一个中心式的critic，但是这样在训练的时候就要求知道global的state和joint action，加入了额外的限制条件。但是用independent训练时，又容易有方差的问题。为了缓解方差的问题，这里的采用非常大的batch size的distributed PPO来做independent的training，因为大的batch包含的东西更多，agent的policy稍微的改变不对对PG造成比较大影响，另外就是比较大的batch能够包含更久的信息，agent不会特别快收敛当前的policy，所以还能帮助探索。

### 有两个网络结构
MLP：两个128个unit的hidden layer（激活函数没有说）

LSTM：
1. 128个unit的全连接，使用RELU
2. 128个hidden state的LSTM
3. 个数为action dimension的全连接

### 策略的细节
使用Gaussian policy，其中mean，diagonal covariance matrix为网络的输出，采样后clipped到合理的control range

在run-to-goal和you-shall-not-pass中使用MLP policy与value function，在sumo和kick-and-defend中使用LSTM policy与value function（下面会提到）

主要的原因是MLP在使用LSTM那两个任务不好。使用LSTM时，我们采用截断的BPTT，在10个timesteps采取截断。

policy与value function采用不同的参数，即不共享参数。

### 另外就是一些训练的小细节：
1. PPO使用clipped objective，$\epsilon=0.2, \gamma=0.995, \lambda=0.95$
2. 每个agent并行做多个rollout，各自优化, collect 409600 samples from the parallel rollouts
3. PPO training in mini-batches consisting of 5120 samples
4. For MLP policies we did 6 epochs of SGD per iteration and for LSTM policies we did 3 epochs
5. don’t use any entropy bonus, 使用L2
6. 每个agent采用4个GPU来训练
7. 像synchronous actor critic，在一整个rollout上计算GAE，因为是在最后才给的reward的
8. 优化器：Adam， lr 0.001
9.  co-efficient $\alpha_t$ in eq. 1 for the exploration reward is annealed to 0 in 500 iterations for all the environments except for kick-and-defend in which it is annealed in 1000 iterations.

# Competitive Environments
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Emergent%20Complexity%20via%20Multi-Agent%20Competition/media/2.png)

### agent的body如下：
1. ant: 12 DoF，8 actuated joints
2. humanoid: 23 DoF，17 actuated joints
形状如上图

### 对于ant body的observations：
all the joint angles of the agent
its velocity of all its joints
the contact forces acting on the body
the relative position and all the joint angles for the opponent. 

### 对于humanoid body的observations：
在ant body的基础上，centre-of-mass based inertia, mass, velocity and the actuator forces for the body 

### 4个环境：
1. run to goal：感觉就是方形的相扑，但是目标是:到达另外一边。先到达的reward+1000，晚到达的reward-1000.如果都没有达到，那么都-1000。
2. you shall not pass: 就是把1的内容稍微改了下，目标是阻止对方到达对面。如果成功阻止了，而且最后成功阻止的agent还站着的话，阻止的agent的reward+1000，如果阻止的agent没有站着的话，它的rewerd为0，然后那个被阻止的agentreward为-1000。如果agent过去了，他的reward为1000，这个阻止失败的agent reward为-1000。
3. sumo：就是相扑，圆形的场地，目标是让对方摔倒或者推出场地。赢的获得+1000的reward，输的获得-1000的reward。如果没有输赢，则都是-1000。
4. kick and defend：射门。一个agent射门，一个agent防守。射门的地方有6个unit宽，成功的射门或者防守获得+1000，失败的获得-1000。防守的agent不能离开goal太远，太远的话，给予-1000的惩罚，另外如果防守成功，而且碰到球，再给予+500的奖励，如果最后还站着，再给+500。

### 对于相扑的环境，还有额外的observations：
torso’s orientation vector 
the distance from the edge of the ring of all the agent
the time remaining in the game 

### 对于点球的环境，还有额外的observations：
the relative position of the ball from the agent
the relative distance of the ball from goal
the relative position of the ball from the two goal posts. 

# Experiments

使用Exploration Curriculum效果更好
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Emergent%20Complexity%20via%20Multi-Agent%20Competition/media/3.png)

不同opponent sampling的影响，ant一直都是均匀的好，因为ant简单，random的防守其实也不错，但是human形状的从一半开始采样好，因为human更难，随机不好
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Emergent%20Complexity%20via%20Multi-Agent%20Competition/media/4.png)

另外为了验证是否over fitting对手策略，这里做了一个实验就是踢球的环境，固定球训练，然后在测试的时候移动球，发现agent不能很好的泛化，只在球的一个位置好：
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Emergent%20Complexity%20via%20Multi-Agent%20Competition/media/5.png)
但是开始一直随机又学不好，因为空间太大，所以这里类时exploration reward的做法，开始小小的随机，随着训练随机形增大


