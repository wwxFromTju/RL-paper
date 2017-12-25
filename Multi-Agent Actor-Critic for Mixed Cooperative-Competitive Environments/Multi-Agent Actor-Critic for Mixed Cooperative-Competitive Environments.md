# Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments

# 核心问题
1. 研究问题：在multiagent设置中，agents同时学习导致环境（学习目标）的不稳定性
2. 假设条件：training时为中心式的训练，能够观察到全部的state $s$，其他agent当前的policy $\pi_i$
3. 主要想法：如果只考虑自己的策略, 那么转移函数$P(s'|s, \pi_i)$是不稳定的（因为状态转移还与其它agent的策略相关），但是如果可以知道全部agent的策略，那么转移函数$P(s'|s, \pi_1,...\pi_n)$是稳定的，所以在学习的时候利用其它agent的策略，能够更有效地对未来做更准确的估计
4. 解决方案：off-line training与on-line testing分开。在off-line training的时候，采用Actor-Critic的架构，每个agent拥有自己的actor与critic，actor只考虑自己的信息，critic能够访问所有信息与所有agent的策略。因为actor只考虑自己的信息，那么在online的时候可以直接使用

## 直接用independent DQN时，replay buffer的问题
但是如果将single agent的DQN直接用到multiagent的环境中, 即每个agent将其它agent视为环境的一部分，memory中只存自己的信息（indepent）：$<s, s', r_i, a_i>$, 因为其它agent的策略在改变，那么memory中体现的$p(s'|s, a_i)$并不一定能反应现在环境，甚至是具有错误的诱导，所以有可能会妨碍agent的学习

举一个我自己理解的例子：
比方说具有合作竞争的两个agent，假设reward只与状态$s$相关，即reward: $r_i(s)$，对于$(s, a_1, a_2)$而言, $r_1(s, a_1) = E_{a_2 \in A, s' \in S}[r_1(s')p(s'|s, a_1, a_2)\pi_2(a_2|s)]$。如果在开始的时候，两个agent都一直合作，那么memory里面的$p(s'|s, a_i)$都是合作的，所以agent能够根据memory估计出一个$r_i(s, a_i)$，但是一旦对手的策略$\pi$改变了，这个估计就错了，所以连$r$的估计都错了，那么就很难学到比较好的policy了

## multiagent时，采用PG可能的问题
这里有一个有意思的结论，在下面的条件下，当只做一次sample的时候，算出来的梯度的方向与真正梯度的方向的夹角小于90的可能性是随着agent数目上升而指数下降的：
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Multi-Agent%20Actor-Critic%20for%20Mixed%20Cooperative-Competitive%20Environments/media/1.png)
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Multi-Agent%20Actor-Critic%20for%20Mixed%20Cooperative-Competitive%20Environments/media/2.png)


大部分都直接可以看出来，其中注意不要有误解:$E(\frac{\hat{\partial}}{\partial{\theta_{i}}} J^2)$其实是$E((\frac{\hat{\partial}}{\partial{\theta_{i}}} J)^2)$
另外就是$\nabla J = <(0.5)^N,(0.5)^N,...(0.5)^N >, \hat \nabla J = <\frac{\hat{\partial}}{\partial{\theta_{1}}} J, \frac{\hat{\partial}}{\partial{\theta_{2}}} J,... \frac{\hat{\partial}}{\partial{\theta_{n}}} J>$，所以想要大于0，只与J的偏导相关，而R只有在a都为1的时候才大于0，那么大于0的概率就等价于所有动作都选1时候的概率


# Methods
## Multi-Agent Actor Critic
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Multi-Agent%20Actor-Critic%20for%20Mixed%20Cooperative-Competitive%20Environments/media/3.png)

在multiagent环境中，因为环境和设置的原因，大多数的设置是agent不能直接知道其他agent的action，policy等信息的，比如在股市中，你并不知道其他人如何买卖股票，但是你可以通过股市的波动来判断买卖的情况

但是如果不知道其他agent的action等信息，又会使的整个环境动态性就比较强（原因是其他agent的策略改变等），实际中为了能够让agent学到相应的行为，policy能够收敛，我们通常会做不同的假设，比如知道对手的action等等

既然在实际中不知道其他agent的信息，但是其他的agent的信息能够很好的帮助学习，很自然的就会想：那么我们就在训练的时候使用这些信息，实际运用的时候的不用这些信息，那就不就很自然地学习出一个更好的agent了（其实这里存在一个问题，除非在训练时将所有可能都学习出来，不然一个固定的policy在面对未知的情况时的性能是未知的，这也是为什么我们需要一个不断学习的agent，而且将所有可能都穷举出来是不太可能的）

更进一步：我们想要在off-line的时候利用更多的信息学习出一个拥有比较好policy的agent，但是为了能够在实际的设置中使用，这个agent的policy的输入与输出在训练与实际使用的时候应该一样，所以无法直接把额外的信息直接结合在policy的输入中，那么有一种想法就是这些额外的信息既然无法直接用，那么就拿来做更准确的梯度的估计，那么很直观的想法就是用AC

主要的原因是：AC分为actor，critic，如果实际使用中不进行训练的话，那么on-line与off-line的共同点就是actor，所以这里的actor我可以设计的尽可能通用，比如只采用自己的observation，$\pi_i(a|o_i)$，然后将额外的信息交给critic，让critic能够帮助policy算出更准确的梯度

所以说从原来的:
$\nabla_{\theta_i}J(\theta_i) = E_{s \sim p^u, a_i \sim \pi_i}[\nabla_{\theta_i}log \pi_i(a_i|o_i)Q_i^\pi(o_i, a_i) ]$变化为：$\nabla_{\theta_i}J(\theta_i) = E_{s \sim p^u, a_i \sim \pi_i}[\nabla_{\theta_i}log \pi_i(a_i|o_i)Q_i^\pi(x,a_1,a_2,....a_n)]$
其中$x=(o_1, o_2, .... o_n)$。通过改变critic Q，我们就可以利用额外的信息帮助agent计算出更稳定的梯度

如果我们采用DDPG的话，那么策略梯度就写为：
$$
\nabla_{\theta_{i}} J(u_i) = E_{x, a \sim D}[\nabla_{\theta_i}u_i(a_i|o_i)\nabla_{a_i}Q_i^u(x, a_1,...,a_n)|_{a_i = u_i(o_i)}]
$$
其中$u_i$代表agent i的策略， D为replay buffer D：$(x, x^1, a_1, . . . , a_N , r_1, . . . , r_N )$
那么critic的更新就写为：
$$
L(\theta_i) = E_{x, a, r, x'}[(Q_i^u(x, a_1,...a_N) - y)^2]\\
y = r_i + \gamma Q_i^{u'}(x', a_1', ..., a_N')|_{a_j' = u_j'(o_j)}
$$
其中$u'$知道的是target policy的参数代表的策略

## Inferring Policies of Other Agents
知道其他agent的策略这个假设其实特别强，所以容易被argue，所以这里弥补一下，放宽假设：知道对手的action，不知道对手的policy。然后通过观察到的action来拟合出对手的policy

所以可以采用极大似然估计来估计policy，另外加上一个entropy让policy不会太确定：
$$
L(\phi_i^j) = -E_{o_j, a_j}[log \hat u^j_i(a_j|o_j) + \lambda H(\hat u_i^j) ]
$$

然后用估计出来的policy来critic更新的时候使用

## Agents with Policy Ensembles
很多时候agent使用的策略只对当前的其他agent使用的策略有效，一旦其他agent稍微变化效果就变差，所以在这里我们对每个agent都训练k个不同的策略，然后在每次play的时候就在这个策略集中随机挑选一个，那么这样就有可能能够学出k个不同的策略，但是在实际中，我们只使用一个policy，所以我们可以利用这k个策略来做权衡，学习出一个总的策略：$J_e(u_i) = E_{k \sim unif(1, K), s \sim p^u, a \sim u_i^{(k)}}[R_i(s, a)]$。对于每个sub policy单独采用MADDPG学习（其实就是为了缓解我说的在实际使用中，其他agent可能会采用不同策略，甚至改变）

