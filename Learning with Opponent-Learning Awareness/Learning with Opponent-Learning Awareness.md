# Learning with Opponent-Learning Awareness

# 核心问题
1. 研究问题：在multiagent learning环境中，training problem non-stationary，而导致的unstable or undesired final results
2. 假设条件：2个agent，agent最大化自己的累积收益，状态s是完全可观察的，在不同设置下分别是：设置1：可以完全访问对手的信息（expected total discounted return V，policy parameter ![](http://latex.codecogs.com/gif.latex?\\theta)） 设置2：可以观察到对手的动作![](http://latex.codecogs.com/gif.latex?u)
3. 主要想法：在做policy gradient update时，考虑对手对于你PG之后的反应，并利用这样的反应来修改PG的梯度
4. 解决方案：在设置1中，相比传统learner，引入一个包含对手Value function的Hessian的二阶项来来考虑对手的策略的变化。然后更进一步，比如不能访问Hessian的梯度的话，在设置1中使用PG来学习的话，将关于Value function变化为对手在轨迹![](http://latex.codecogs.com/gif.latex?\\tau)下的total discounted reward与策略。在设置2中，利用观察到的对手的action，对于对手的策略采用maximum likelihood建立opponent modeling，然后同设置1的做法来学习


more detial，在multiagent learning环境中，agent进行学习而不断改变策略。因为agent的最优策略与其他agent的策略相关，在学习过程中agent间相互影响，agent可能会因为对手的策略改变而不断改变自己的策略，使的training problem non-stationary，而导致的unstable or undesired final results

# Methods
与single agent类似，multiagent的主要目的还是最大化累积reward。在multiagent中，agent间可能存在竞争或者合作的关系，这样的关系可以在reward中直接体现，比如竞争的关系：一者reward上升，另外一者reward就下降，或者合作的关系：两个agent获得相同的reward。相比single agent简单地最大化自己的累积reward不同，在multiagent的设置中，需要针对不同的任务来提出相应的方法

## Naive Learner
不论竞争还是合作，一种最简单的处理办法就是：类似single agent的做法，直接最大化自己的reward。当然在合作任务中，这样的做法并不一定有效，但是通常也有一定的效果

假设agent1的policy ![](http://latex.codecogs.com/gif.latex?\\pi^1)是由参数![](http://latex.codecogs.com/gif.latex?\\theta^1)表示，那么agent 1的expected total discounted return就可以写成![](http://latex.codecogs.com/gif.latex?V^1(\\theta^1, \\theta^2))，那么同样我们可以写出agent2的policy ![](http://latex.codecogs.com/gif.latex?\\pi^2)，expected total discounted return ![](http://latex.codecogs.com/gif.latex?V^2(\\theta^1, \\theta^2))

假设在时刻![](http://latex.codecogs.com/gif.latex?t)时，agent1的策略为![](http://latex.codecogs.com/gif.latex?\\pi^1_t), expected total discounted return为![](http://latex.codecogs.com/gif.latex?V^1_t(\theta^1, \theta^2)),agent2的策略为![](http://latex.codecogs.com/gif.latex?\\pi^2_t), expected total discounted return为![](http://latex.codecogs.com/gif.latex?V^2_t(\\theta^1, \\theta^2))

那么在时刻![](http://latex.codecogs.com/gif.latex?t)时，我们可以直观地利用当前的局面(即对手的策略)来选择下一个时刻![](http://latex.codecogs.com/gif.latex?t+1)的策略，形式化地写成：
![](http://latex.codecogs.com/gif.latex?\\\\
\\theta^1_{t+1} = argmax_{\\theta^1} V^1(\\theta^1, \\theta_t^2)\\\\
\\theta^2_{t+1} = argmax_{\\theta^2} V^1(\\theta_t^1, \\theta^2))
这里存在一个问题，当V的空间是无限大的时候（比如action是连续的），那么很多时候我们并不能直接通过求解上面的方程组来获得下一个时刻![](http://latex.codecogs.com/gif.latex?t+1)的参数![](http://latex.codecogs.com/gif.latex?\\theta_{t+1})。一种很直观的想法当然是通过求解梯度的方法来不断迭代：
![](http://latex.codecogs.com/gif.latex?\\\\
\\theta^1_{t+1} = \\theta^1_t + f^1_{nl}(\\theta^1_t, \\theta^2_t)\\\\
f^1_{nl} = \\frac{\\partial V^1(\\theta^1_t, \\theta^2_t)}{\\partial \\theta^1_t} \\cdot \\delta)
其中，![](http://latex.codecogs.com/gif.latex?\\delta)为step size

## Learning with Opponent Learning Awareness
Naive Learner的基本假设是：因为你的求解或者迭代是假设对手的策略是固定的，存在一个很直接的问题：你在学，别人也在学，那么你学的并不一定有效

很自然，我们就会思考，比如把对手也在学习的这一部分信息考虑进来是不是对于agent的学习有帮助呢？

LOLA（Learning with Opponent Learning Awareness）就加入了这部分的考虑，将优化的目的修改为：
![](http://latex.codecogs.com/gif.latex?\\\\
\\theta^a_{t+1} = \\theta^a_{t} + \\Delta\\theta^a, a\\in\\{0,1\\} \\\\
where \\\\
\\Delta\\theta^1 = \\mathop{\\arg\\max}_{\\Delta\\theta^1:||\\Delta\\theta^1|| \\leq\\delta}V^1(\\theta^1_t + \\Delta\\theta^1, \\theta^2_t + \\mathop{\\arg\\max}_{\\Delta\\theta^2:||\\Delta\\theta^2|| \\leq\\delta}V^2(\\theta^1_t + \\Delta\\theta^1, \\theta^2_t + \\Delta\\theta^2 ) ) \\\\
\\Delta\\theta^2 = \\mathop{\\arg\\max}_{\\Delta\\theta^2:||\\Delta\\theta^2|| \\leq\\delta}V^2(\\theta^1_t + \\mathop{\\arg\\max}_{\\Delta\\theta^1:||\\Delta\\theta^1|| \\leq\\delta}V^1(\\theta^1_t + \\Delta\\theta^1, \\theta^2_t + \\Delta\\theta^2 ), \\theta^2_t + \\Delta\\theta^2))

与上面类似，action的空间可能是无限的，无法一一访问求解，同样我们将其修改为梯度的方法：
![](http://latex.codecogs.com/gif.latex?\\\\
\\theta^1_{t+1} = \\theta^1_t + f^1_{lola}(\\theta^1_t, \\theta^2_t)\\\\
f^1_{lola} = \\frac{\\partial V^1(\\theta^1_t, \\theta^2_t)}{\\partial \\theta^1_t} \\cdot \\delta + (\\frac{\\partial V^1(\\theta^1_t, \\theta^2_t)}{\\partial \\theta^2_t})^T\\frac{\\partial^2V^2(\\theta^1_t, \\theta^2_t)}{\\partial \\theta^1_{t} \\partial\\theta^2_t} \\cdot \\delta \\eta)

其中 ![](http://latex.codecogs.com/gif.latex?\\delta)是一阶的step size， ![](http://latex.codecogs.com/gif.latex?\\eta)是二阶的step size

论文上没有从方程变化为迭代的方法的推导过程, 请教一下张程伟学长，我自己总结一下，因为我们的目的是：
![](http://latex.codecogs.com/gif.latex?\\\\
\\Delta\\theta^1 = \\mathop{\\arg\\max}_{\\Delta\\theta^1:||\\Delta\\theta^1|| \\leq\\delta}V^1(\\theta^1_t + \\Delta\\theta^1, \\theta^2_t + \\mathop{\\arg\\max}_{\\Delta\\theta^2:||\\Delta\\theta^2|| \\leq\\delta}V^2(\\theta^1_t + \\Delta\\theta^1, \\theta^2_t + \\Delta\\theta^2 ) ))
转化为梯度的方法的话，其实目的就是求解梯度的方向，从极限的角度来理解的话，以agent1为例，那么我们可以假设![](http://latex.codecogs.com/gif.latex?\\Delta\\theta^1 \\rightarrow 0), 所以可以近似地看成对下面的式子求解梯度方向：
![](http://latex.codecogs.com/gif.latex?\\\\
V^1(\\theta^1_t, \\theta^2_t + \\mathop{\\arg\\max}_{\\Delta\\theta^2:||\\Delta\\theta^2|| \\leq\\delta}V^2(\\theta^1_t, \\theta^2_t + \\Delta\\theta^2 ) )
)
进一步，把![](http://latex.codecogs.com/gif.latex?V^2)写成迭代的形式：
![](http://latex.codecogs.com/gif.latex?\\\\
V^1(\\theta^1_t, \\theta^2_t + \\frac{\\partial V^2(\\theta^1_t, \\theta^2_t)}{\\partial \\theta^2_t} \\cdot \\delta)
)
那么它的梯度为：
![](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Cfrac%7B%5Cpartial%20V%5E1%28%5Ctheta%5E1_t%2C%20%5Ctheta%5E2_t%20&plus;%20%5Cfrac%7B%5Cpartial%20V%5E2%28%5Ctheta%5E1_t%2C%20%5Ctheta%5E2_t%29%7D%7B%5Cpartial%20%5Ctheta%5E2_t%7D%20%5Ccdot%20%5Cdelta%29%7D%7B%5Cpartial%20%5Ctheta%5E1_t%7D%20%5Capprox%20%5C%5C%20%5Cfrac%7B%5Cpartial%20V%5E1%28%5Ctheta%5E1_t%2C%20%5Ctheta%5E2_t%29%7D%7B%5Cpartial%20%5Ctheta%5E1_t%7D%20&plus;%20%5Cfrac%7B%5Cpartial%20V%5E1%28%5Ctheta%5E1_t%2C%20%5Ctheta%5E2_t%29%7D%7B%5Cpartial%20%5Ctheta%5E2_t%7D%20%5Ccdot%20%5Cfrac%7B%5Cpartial%28%5Cfrac%7B%5Cpartial%20V%5E2%28%5Ctheta%5E1_t%2C%20%5Ctheta%5E2_t%29%7D%7B%5Cpartial%20%5Ctheta%5E2_t%7D%20%5Ccdot%20%5Cdelta%29%7D%7B%5Cpartial%20%5Ctheta%5E1_t%7D%20%3D%20%5C%5C%20%5Cfrac%7B%5Cpartial%20V%5E1%28%5Ctheta%5E1_t%2C%20%5Ctheta%5E2_t%29%7D%7B%5Cpartial%20%5Ctheta%5E1_t%7D%20&plus;%20%28%5Cfrac%7B%5Cpartial%20V%5E1%28%5Ctheta%5E1_t%2C%20%5Ctheta%5E2_t%29%7D%7B%5Cpartial%20%5Ctheta%5E2_t%7D%29%5ET%20%5Ccdot%20%5Cfrac%7B%5Cpartial%5E2%20V%5E2%28%5Ctheta%5E1_t%2C%20%5Ctheta%5E2_t%29%29%7D%7B%5Cpartial%20%5Ctheta%5E1_t%20%5Cpartial%20%5Ctheta%5E2_t%7D%20%5Ccdot%20%5Cdelta)
那么最后乘上step size就近似为：
![](http://latex.codecogs.com/gif.latex?%5Cinline%20f%5E1_%7Blola%7D%20%3D%20%5Cfrac%7B%5Cpartial%20V%5E1%28%5Ctheta%5E1_t%2C%20%5Ctheta%5E2_t%29%7D%7B%5Cpartial%20%5Ctheta%5E1_t%7D%20%5Ccdot%20%5Cdelta%20&plus;%20%28%5Cfrac%7B%5Cpartial%20V%5E1%28%5Ctheta%5E1_t%2C%20%5Ctheta%5E2_t%29%7D%7B%5Cpartial%20%5Ctheta%5E2_t%7D%29%5ET%5Cfrac%7B%5Cpartial%5E2V%5E2%28%5Ctheta%5E1_t%2C%20%5Ctheta%5E2_t%29%7D%7B%5Cpartial%20%5Ctheta%5E1_%7Bt%7D%20%5Cpartial%5Ctheta%5E2_t%7D%20%5Ccdot%20%5Cdelta%20%5Ceta)

## Learning via Policy Gradient
当不可以exact gradients of Hessians时候，也就是无法访问V的时候，我们可以采用在一条轨迹的![](http://latex.codecogs.com/gif.latex?R_t^a(\\tau) = \\sum^T_{l=t}\\gamma^{l-t}r_l^a)来替代![](http://latex.codecogs.com/gif.latex?V^a)的作用，其中![](http://latex.codecogs.com/gif.latex?\\tau = (s_0, u_0^0, u_0^1, r_0^0, r_0^1, s_1...u_t^0, u_t^1 \\, r_t^0, r_t^1))

对于navie learner, 以agent1为例我们可以写成：
![](http://latex.codecogs.com/gif.latex?%5Cinline%20f_%7Bnl%2C%20pg%7D%5E%7B1%7D%20%3D%20%5Cnabla_%7B%5Ctheta%5E1%7D%20%5Cmathbb%7BE%7D%28R%5E1_0%28%5Ctau%29%29%20%5Ccdot%20%5Cdelta)

其中：
![](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Cnabla_%7B%5Ctheta%5E1%7D%20%5Cmathbb%7BE%7D%28R%5E1_0%28%5Ctau%29%29%20%3D%20%5Cmathbb%7BE%7D%5BR%5E1_0%28%5Ctau%29%5Cnabla_%7B%5Ctheta%5E1%7Dlog%5Cpi%5E1%28%5Ctau%29%5D%20%5C%5C%20%3D%20%5Cmathbb%7BE%7D%5B%5Csum%5ET_%7Bt%3D0%7D%5Cnabla_%7B%5Ctheta%5E1%7Dlog%5Cpi%5E1%28u%5E1_t%7Cs_t%29%5Cgamma%5Et%28%20R%5E1_0%28%5Ctau%29%20-%20b%28s_t%29%29%5D)

对于LOLA，我们可以写成：
![](http://latex.codecogs.com/gif.latex?%5Cinline%20f_%7Blola%2C%20pg%7D%5E%7B1%7D%20%3D%20%5Cnabla_%7B%5Ctheta%5E1%7D%20%5Cmathbb%7BE%7D%28R%5E1_0%28%5Ctau%29%29%20%5Ccdot%20%5Cdelta%20&plus;%20%28%5Cnabla_%7B%5Ctheta%5E2%7D%5Cmathbb%7BE%7DR_0%5E1%28%5Ctau%29%29%5ET%5Cnabla_%7B%5Ctheta%5E1%7D%5Cnabla_%7B%5Ctheta%5E2%7D%5Cmathbb%7BE%7DR_0%5E2%28%5Ctau%29%20%5Ccdot%20%5Cdelta%5Ceta%20%5C%5C)


![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Learning%20with%20Opponent-Learning%20Awareness/media/1.png)

## LOLA with Opponent Modeling
在Learning via Policy Gradient中，虽然放宽了条件：不需要访问对手的V，但是需要访问对手的policy ![](http://latex.codecogs.com/gif.latex?\\pi)。这个假设其实还是很强的，所以自然地，我们想要将这个强的限制放宽，但是算法本身是需要对手的信息（即对手的策略的），所以这里我们可以利用对手以前在状态![](http://latex.codecogs.com/gif.latex?s)下的action ![](http://latex.codecogs.com/gif.latex?a) 对对手建模，也就是设置2

将策略看成概率分布，那么就是一个估计的问题，自然会想到用maximum likelihood来估计：![](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Chat%7B%5Ctheta%7D%20%3D%20%5Cmathop%7B%5Carg%5Cmax%7D_%7B%5Ctheta%7D%5Csum_t%20log%20%5Cpi_%7B%5Ctheta%7D%28u_t%7Cs_t%29)

## Higher Order LOLA
并没有详细说明，就是说可以考虑用更高阶来做LOLA，我觉得应该就是考虑得更多步吧

# Experimental Setup
## Iterated Games
可以简单地理解为：在每个状态![](http://latex.codecogs.com/gif.latex?S_t)下都是同样的局面，能够采取的action一样，获得的reward一样。

the iterated prisoners dilemma (IPD)为例，存在2个agent，每局都是面对prisoners dilemma，agent能够采取的动作为C或D，并获得相应的reward，然后进入下一局，面对同样的局面。
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Learning%20with%20Opponent-Learning%20Awareness/media/2.png)

与single-shot prisoners’ dilemma相比，IPD中是可以考虑对手的行为的并调整自己的策略，比如对手是可以合作的，那么我们可以诱导对手合作，并利用之前play的局面来调整策略。但是在single-shot prisoners’ dilemma中我们无法对对手做出假设，那么最佳的方案当然是D

在single-shot prisoners’ dilemma中存在的均衡为：（D，D），在IPD中存在的很多均衡，比如al- ways defect strategy (DD)和tit-for-tat (TFT)， DD就是一直采用D，TFT就是在开局的时候采用合作，在以后都采用对手上局采用的行为。对于self-play时，DD每个step的平均return是-2，TFT的平均return是-1

上面是可以存在纯策略的均衡，下面的iterated matching pennies (IMP)，类似IPD，但是只有混合策略的均衡：0.5 选 head， 0.5 选 tail
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Learning%20with%20Opponent-Learning%20Awareness/media/3.png)

对于上述的2个game，我们可以把它们model成two-agent MDP，状态![](http://latex.codecogs.com/gif.latex?s_t = (u^1_{t-1}, u^2_{t-1}))，所以一共有5个state：s0, (C, C), (C, D), (D, C), (D, D)。

因为这样的空间很小，所以能够简单地算出future discounted reward，所以我们可以calculate the exact policy update for both NL and LOLA agents.

同时这里assume that agents can only update their poli- cies between the rollouts, not during the iterated game play.

## Coin Game
Coin Game是IPD在higher dimensional，multi-step actions上的扩展，细节如图所示，需要补充的是：当一枚硬币被捡起来会随机出现颜色随机的另外一枚硬币。所以agent需要考虑recurrent policies和features sequential actions.
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Learning%20with%20Opponent-Learning%20Awareness/media/4.png)

greedy的agent就是不管什么颜色都去捡起来，那么大家的平均reward都是0。相对合作的策略就是：各自捡自己颜色，这样的平均reward是0.5

在这个环境中，LOLA和NL采用agents’ policies are parametrized with a recurrent neural network，同时也不好算future discounted reward，所以LOLA用了opponent-modelling

## Training Details
* gradient descent with step size 0.005 for the actor
* gradient descent with step size 1 for the actor
* batch size 4000.
* ![](http://latex.codecogs.com/gif.latex?\\gamma) is set to 0.96 for the prisoners’ dilemma and the coin game
* ![](http://latex.codecogs.com/gif.latex?\\gamma) is set to 0.9 for matching pennies

对于coin game：
输入为4个channel的grid，其中用2个channels表示两个agent的位置，用2个channels表示red coins，blue coins

网络结构
2个卷积层
* 3 × 3 filters
* stride 1
* relu
然后在32个hidden units recurrent neural network

# Results

##Iterated Games
### IPD
NL的方法最后几乎都学到DD，但是LOLA可以学到TFT
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Learning%20with%20Opponent-Learning%20Awareness/media/5.png)

### IMP
NL学的很混乱，几乎学不到东西。LOLA可以学到快Nash均衡了，0.5选head，0.5选tail
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Learning%20with%20Opponent-Learning%20Awareness/media/6.png)

### coin game
LOLA学到了
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Learning%20with%20Opponent-Learning%20Awareness/media/7.png)

