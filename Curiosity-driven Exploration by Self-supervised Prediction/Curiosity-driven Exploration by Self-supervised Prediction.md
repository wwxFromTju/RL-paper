# Curiosity-driven Exploration by Self-supervised Prediction

# 核心问题
1. 研究问题：在稀疏的reward或者几乎没有reward的环境中，如何让agent更有效地对环境进行探索，甚至学习到一些技能？
2. 假设条件：在video game（视频游戏）中，可以控制出生的位置，环境墙壁的纹理，不同的关卡
3. 主要想法：利用'好奇心'作为内在的reward信号来让agent更有效地进行探索，甚至学习到技能
4. 解决方案：将’好奇心‘的reward建模成agent对于新状态在visual feature space中的预测与实际状态在visual feature space的表征的不同，同时利用self-supervised inverse dynamics model来帮助agent对特征空间进行状态特征的提取，并在面对新环境中进行fine-tuning


# Curiosity-Driven Exploration
在传统的RL形式中，将原来的reward信号，改写成$r_t = r_t^i +r_t^e$，其中$r_i^t$表示由于好奇心带来的内在的奖励，$r_i^e$表示环境中本来就有的奖励，比如达到目标获得奖励。然后再采用传统的RL算法，通过最大化自己的累积收益来学习到相应的策略。

这里的$r_i^e$可能是稀疏的，甚至没有的。所以希望通过$r_i^t$在稀疏$r_i^e$的时候，帮助agent更少与环境交互，更快地学习到相应的策略。另外在没有$r_i^e$，利用$r_i^e$来让agent更有效地探索环境。

那么一个很重要的问题就是：designing an intrinsic reward signal based on prediction error？

## Prediction error as curiosity reward
设计intrinsic reward很大程度上与你想要解决的任务相关，这里想要解决的问题是video game，那么就要立足在video game获得state就是一张图片。一种很直接的想法就是：在状态$s$下，采用动作$a$，通过卷积与反卷积等结构来预测下一状态$s'$，通过预测出来的$s'$与实际的$s'_r$的误差作为agent的reward。

直接预测下一个状态$s'$其实存在一系列问题，比如预测每个pixel的颜色是一件很困难的事，稍微有些差别，prediction error就会变化的比较大，所以agent很容易被这样的信息误导，并没有达到探索的目的。此外图片中的信息其实非常丰富，比如在不同关卡的背景，亮度不同，但是本质的内含相近，关注这些额外的信息反而会影响在不同关卡中的泛化。

其实直接预测图片带来的问题，本质上就是：在state space中可以分成三种信息，1.agent可以控制，比如agent开枪射出子弹，2.agent不可以控制的，但是对agent有实际影响的，比如怪物的移动，3.本质上无效的信息，比如agent走着走着，从白天变成黑夜，背景变黑了。

那么其实我们真正关注的是第1和第2种信息，因为这2种信息才是本质上影响agent决策的信息，那么我们的curiosity reward应该建立在这2种信息上才会让agent更有效的探索学习。

## Self-supervised prediction for exploration
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Curiosity-driven%20Exploration%20by%20Self-supervised%20Prediction/media/1.png)

那么想要利用第1和第2种信息，本质上就是对state做特征提取，很自然就会想到利用神经网络来手工完成特征，同时我们希望提出的特征是由第一类，第二类信息影响的。但是第二类信息不由agent控制，所以就可以focus住在第一类信息中：由agent的action影响的特征。

那么很直观的就是：利用当前状态$s$与下一状态$s'$来预测采用了什么样的动作$a$。

在这个核心思想的指导下，我们可以设计一个神经网络对状态$s$做特征抽取$\phi(s)$，并利用下一状态的特征抽取$\phi(s')$，来预测在这两个状态间采用的action：
$$
a_{prediction} = DNN(s, s';\theta_I) 
$$
通过最小化预测的$a_{prediction}$与实际采用动作$a$的误差，利用反向传播让神经网络学习到真正由action影响的特征。

因为这里的action是离散的，所以我们可以对预测的action做soft-max，然后通过maximum likelihood estimation来设置对应的loss函数,即
$$
\min_{\theta_I}L_I(a_{prediction}, a)
$$

上面说了很多，本质上就是怎么能够提取出真正由agent影响的因素，但是不要忘记我们真正的目的是利用对于状态$s'$的prediction error来建模agent的curiosity。所以在提取出特征$\phi(s)$后，我们同样可以使用神经网络来预测下一个状态$s'$的特征$\hat{\phi(s')}$:
$$
\hat{\phi(s')} = f(\phi(s), a;\theta_F
$$

因为预测出来的特征是个向量，而且我们也不清楚向量中的每个元素代表什么，所以一个很直观的想法就是利用$L_2$范数来作为loss：
$$
L_F(\phi(s'), \hat{\phi(s)}) = \frac{1}{2} \big\| \phi(s') - \hat{\phi(s)} \big\|^2_2 
$$

同时，我们可以利用上面的损失$L_F(\phi(s'), \hat{\phi(s)})$来构造curiosity reward：$r_t^i = \eta L_F(\phi(s'), \hat{\phi(s)})$

最后我们可以将agent的学习目标定成：
$$
\min_{\theta_P, \theta_I, \theta_F} \big[-\lambda\mathbb{E}_{\pi(s_t;\theta_P)}[\sum_tr_t] + (1 - \beta)L_I + \beta L_F \big]
$$
其中$1 \leq \beta \leq 1, \lambda > 1 $，只是作为相应项的尺度的衡量。

那么在训练的时候，因为不断对$\theta_F$做优化，所以最开始部分预测准确之后，为了获得更多的curiosity reward就会主动地去探索更多的未知的状态（其实本质就在这里）

# Experimental Setup
## 环境
第一个环境是Doom：
reward很稀疏，只有在到达goal之后，会给予+1的reward。agent有四种action：move forward，move left，move right，no-action。每个episode最多2100步。整张map一共有9个房间，有三种出生方式：
1. dense的reward：在地图上任意蓝点出生
2. sparse：在room 13出生
3. very sparse：在room 17出生
同时pre-train的map是在另外一张地图上，具有不同的纹理。
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Curiosity-driven%20Exploration%20by%20Self-supervised%20Prediction/media/2.png)

![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Curiosity-driven%20Exploration%20by%20Self-supervised%20Prediction/media/3.png)



第二个环境是Super Mario Bros：
一共有4个level，在第一个level上做pre-train，然后看在其他level上的泛化程度。In this setup, we reparametrize the action space of the agent into 14 unique actions following。

## Training details
将图片gray-scale，然后re-size到42x42上，然后把最近的4个frame叠起来，在训练的时候采用action repeat，在Doom中action保持4个frame，在Mario中action保持6个frame。但是测试的时候，不采用action repeat。采用A3C，使用20个worker，SGD，ADAM， parameters not shared across the workers.

## A3C网络结构
4层卷积层，每层：
 * filters: 32
 * kernel size: 3x3
 * stride: 2
 * padding: 1
 * 激活函数：ELU

接256个unit的LSTM层， 然后分叉为2个全连接，分别作为value function与action function

## Intrinsic Curiosity Module (ICM) architecture
The inverse model first maps the input state $s$ into a feature vector $\phi(s)$，输出的$\phi(s)$维度为288：
4层卷积，每层：
 * filters: 32
 * kernel size: 3x3
 * stride: 2
 * padding: 1
 * 激活函数：ELU
然后将状态$\phi(s)$与$\phi(s')$拼接起来，接一个256个unit的全连接层，然后接4个unit的全连接层作为action的输出

The forward model：
将$\phi(s)$与$a$拼接，然后接256个unit的全连接层，然后再接288个unit的全连接层，输出作为下一个状态$\hat{\phi(s')}$的预测

我们将$\beta = 0.2, \lambda=0.1, lr=1e-3$

## 对比方法
1. ICM（pixels） + A3C
2. TRPO + VIME

# Experiments
##  Sparse Extrinsic Reward Setting
reward之后在到达goal才获得，所以是sparse的。在这里我们采用三种初始化方式（在上面提到过），因为这三种初始化方式到达goal的难度不一样，所以我们进一步分成了三种境况（上面提到过的）

总的而言，有ICM比没有ICM的好，有ICM能够更快地的探索到目标。在“Dense”中，因为初始状态不一样，因为房间纹理会不一样，agent用的ICM（pixels）带了上面提到的问题（agent不能控制，也不影响agent的因素），所以学的慢。

另外在“sparse”的情况下，pixles有段时间最好，这里解释为：因为在同一个房间内初始化，所以纹理固定，所以agent能够学的快一些（ICM的话，因为要额外训练提取，所以会慢一些）。

在“very sparse”的情况下，只有ICM好，说明提取特征的确有用。
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Curiosity-driven%20Exploration%20by%20Self-supervised%20Prediction/media/4.png)


### Robustness to uncontrollable dynamics
就是说明ICM对于一些不能控制的因数的鲁棒性，将原来的输入固定在图片一角，然后其他用的white noise，这样的话，ICM（pixles）应该会被white noise吸引，导致学不好。然后重新在‘sparse’的情况下来做实验（就是上面ICM比ICM（pixels）慢的），然后这边发现ICM比ICM（pixles）快了。
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Curiosity-driven%20Exploration%20by%20Self-supervised%20Prediction/media/5.png)

![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Curiosity-driven%20Exploration%20by%20Self-supervised%20Prediction/media/6.png)


### Comparison to TRPO-VIME
就是和别的算法比较，VIME：variational information maximization
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Curiosity-driven%20Exploration%20by%20Self-supervised%20Prediction/media/7.png)


## No Reward Setting
因为没有reward信号，所以其实我们不知道该学什么，所以这里主要判断的是在没有reward信号的时候，agent的探索能力。

一个好的exploration policy就是能够让agent访问过经可能多的状态，因为一个随机的探索很可能只让agent在出身点附近打转，而不会探索到远的地方。

### DOOM
前三张图为采用ICM的探索过的房间，后面2个是普通探索。发现ICM探索得更多。
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Curiosity-driven%20Exploration%20by%20Self-supervised%20Prediction/media/8.png)


### Mario
Mario agent can learn to cross over 30% of Level-1.这里比较有趣的是，所以没有给agent杀死怪物的reward，但是agent学会了杀死怪物，主要的猜想是说因为碰到怪物就死了，所以没有更多的好奇的reward，为了最大化好奇的reward，所以agent学习到了杀死怪物

## Generalization to Novel Scenarios
为了研究agent是真学到了，还是只是记住当前环境，这里在Mario的level-1中，没有reward，只有ICM进行学习，然后将agent分为三类：
1. 在level 1中学好的agent直接用别的level中
2. 在当前level中从零开始学
3. 在level 1中学好的agent，还是没有reward，只有ICM，然后在当前level继续学
4. 在level 1中学好的agent，有reward了，同时有ICM，在当前的level继续学
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Curiosity-driven%20Exploration%20by%20Self-supervised%20Prediction/media/9.png)
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Curiosity-driven%20Exploration%20by%20Self-supervised%20Prediction/media/10.png)



这里发现，在level 2的时候，‘直接用’的agent效果差。分析了一下，主要是level 1和level 3的环境都是白天，level 2的环境是晚上，所以在level 1学完在level 3直接用还好，但是level 2因为环境不一样，所以效果差。

另外在level 2与level 3中，从头开始学的agent效果差，主要的原因是这两个level 1更难，所以更难学到。

另外在level 3中，利用ICM和有外在reward再继续学都差，level 3难，所以学不到好的policy，同时导致以前学习的被最新的policy覆盖

此外有个结论就是，通过ICM预训练，然后再将reward加上，能够更快地学习到相应的policy

