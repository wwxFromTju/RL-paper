# Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments

# 核心问题
1. 研究问题：如何在nonstationary的环境中，快速的学习出相应的策略（这里的Competitive是导致Competitive的一种原因）
2. 假设条件：nonstationary的环境是由多个stationary的task切换引起的，任务切换具有Markov chain的性质
3. 主要想法：找到一个默认参数$\theta$，通过在任务$T_i$的更新，能够在新任务$T_{i+1}$上有好的效果
4. 解决方案：修改MAML中关于loss的定义，在任务$T_i$来采样，调整原始策略$\pi_\theta$, 将评估新策略$\pi_\phi$放在新的任务$T_{i+1}$上

# background
当前RL做的比较好的领域，包括：playing games，robotics，dialogue systems等方面，目前效果较好的算法中都有一个最基本的假设：环境是stationary，所以可以针对固定的环境来训练出一个固定的策略，在online的时候使用这个策略，而不需要调整。

但是在real world中，正如：“人不能踏入同一条河流”，环境经常是nonstationary。比如在multiagent的环境中，如果我们把其他agent看成环境的一部分，如果其他agent为了更高的收益调整自己的policy，从agent的角度来看，是因为环境的变化而导致到了自己policy的效果的变化，即其他agent的变化导致了multiagent的nonstationary。（其他方面比如：gent的特征发生改变（比如在robot的设置中，其他agent改变成不同的robot）等等）

当前存在一些解决nonstationary中learning的方法，比如：context detection，tracking等，更具体的比如：针对环境的changes，不断的对policy做fine-tuning等等，但是这些方法有效的基本假设是：policy能够快速改变。

所以上面的解决nonstationary或许能够在传统基于表格等的RL算法上有比较好的效果，但是直接运用在基于NN的DRL算法，可能会存在一些问题。因为DRL是通过BP来微调网络参数的，网络参数牵一发而动全身，为了policy的效果和稳定性，需要相对多的data，即存在sample inefficient的问题。但是在nonstationary的环境中，反应当前环境的动态性的数据应该是最近期的数据，离当前越远，相关性越差，即我们能够获得只有一小部分关于当前环境动态的data，这就与sample inefficient存在矛盾。其实我觉得nonstationary的环境最难的就是：在少量的数据中，马上根据环境来调整自己的policy。

所以想要直接结合原来context detection等方法与DRL，必然存在问题。

# 思路
可以将nonstationary的环境看成由sequence of stationary task组成的，然后就可以看成一个multi-task learning problem。这里的主要的假设出发点，我觉得是：nonstationary不能变化太剧烈，太剧烈就不适用于用DRL来解决，所以我们可以假设变化的稍微缓慢一些，因为变化缓慢，在一定的时间内，我们甚至可以视为环境没有变化，只有在超过一定时间时，我们认为环境变化了。所以在一定时间范围内，我们可以视为是一个task，在更大的时间尺度下，是多个task间的切换。

既然可以建模成多个task间的切换，必然就会想到解决多个task的MAML算法（能够在few-shot regime中，通过少量的数据就产生flexible learning rules）。

这里，我们需要解决的是：task之间切换的问题，即在前一时刻$T_i$的任务下来调整策略 $\pi_t$，然后获得在task $T_{t+1}$下表现好 $\pi_{t+1}$。

所以我们必然需要考虑task $T_{t}$与task $T_{t+1}$之间的关系，通过上面sequence of stationary task那部分我的想法，这里可以很直观地建模成一个markov chain。所以我们就可以稍微修改一下MAML中关于loss的定义，来找到能通过task $T_i$就能够调整出策略 $\pi_\phi$，在新任务中有相对好的效果。

# 具体做法
###MAML
这里居于之前提出的model-agnostic meta-learning（MAML），扩展到dynamically changing task中。所以我先简单地介绍一下MAML的做法。

MAML中将每个任务T定义为：$(L_T, P_T(x), P_T(x_{t+1}|x_t, a_t), H)$, 其中$L_T$是根据特定的task设计的loss，$P_T(x)$第一个状态$x$的概率，$P_T(x_{t+1}|x_t, a_t)$是相应的状态转移函数，$H$定义了horizon（即计算loss时候考虑的trajectory的长度）。更具体的是：在RL中，我们可以将$L_T$定义为：$-\sum_t^HR_T(x_t, a_t)$，注意一下这里不是discount reward的定义（所以我们要定义H）

假设我们已经训练好了一个默认参数$\theta$，即$\pi_{\theta}$，通过$\pi_{\theta}$在task $T$上采样出K条trajectories $\tau_\theta^{1:K}$，然后通过GD来调整：
$$
\phi = \theta - \alpha \bigtriangledown_\theta L_T(\tau_\theta^{1:K}),\\ where\ L_T(\tau_\theta^{1:K}) = \frac{1}{K} \sum_{k=1}^K L_T(\tau_{theta}^{k})
$$

因为我们的目的是在多个task上通过GD来获得特定任务的策略$\pi_\phi$，所以我们应该是在所有任务上$D(T)$来找到一个比较好的初始参数$\theta$来调整，所以我们能够定义：
$$
min_{\theta}\ E_{T\sim D(T)}[R_T(\theta)]\\
 where\ R_{T}(\theta) = E_{\tau_\theta^{1:K} \sim P_T(\tau|\theta)}[E_{\tau_\phi\sim P_T(\tau|\phi)}[L_T(\tau_\phi)|\tau_\phi^{1:k}, \theta]]
$$
这个最小的等式的直观解释是：我们希望能找到一个$\theta$，这个$\theta$在某个（新）任务task $T$上采样K条trajectories，然后做梯度下降得到$\phi$，这个$\pi_\phi$能够任务task $T$上的到好效果，即：用$\phi$对T也做采样出trajectories，然后计算在这个trajectories上面的loss。然后我们找一个能在所有task下，loss期望（求和）最小的，即在所有任务中（求和）loss最小的。其中$L_T(\tau_\phi)|\tau_\phi^{1:k}, \theta$为条件的意思，即$\phi$是居于$\theta$与trajectory $\tau_\phi^{1:k}$来计算的。

对于上面的min的式子，在实际中，我们可以写成：
$$
\bigtriangledown_{\theta}\ R_{T}(\theta) = E_{\tau_\theta^{1:K} \sim P_T(\tau|\theta),\tau_\phi\sim P_T(\tau|\phi)}[L_T(\tau_\phi)[\bigtriangledown_{\theta}log \pi_{\phi}(\tau_\phi) + \bigtriangledown_{\theta}\sum_{k=1}^{K}log\pi_{\theta}(\tau_\theta^k)]]
$$
具体的求导过程在附录中

### continue learning
在最初的MAML的设置中，我们关注的是在不同task之间的共同属性，通过对共同属性的求解，来获得一个比较好的初始化参数，然后根据特定的task来微调参数，希望在task上有好的效果。然后所谓连续学习，必然是会考虑时间上先后出现task之间存在一定的关系，更具体的是：$T_{i-1}$应该包含了关于$T_i$的信息。那么我们就能够通过$T_{i-1}$的信息来调整参数，希望在$T_{i}$上能够有比较好的效果。

比方在multiagent的环境中，其他agent的policy可能会随着时间而不断变化，但是这样的policy的变化并不是无迹可寻，而是按照自己之前play获得信息和reward来调整的，所以我们如果把之前round看成一个task，然后在agent调整好policy后的下一个round就可以看成下一个task。

所以对不停调整的task而言，我们的目的是一堆的task转移之间，最大自己的累积收益，直观上就是
$$
min_{\phi_{1}}\ E_{P(T_0), P(T_{i+1}|T_i))}[R_{T_0, T_1, ...T_{L}}(\phi_{i})]
$$
上面的式子意思就是说，找到一个最佳的初始参数$\phi_0$,在一系列的task中，不断根据上一个task T的参数$\phi_{i}$来调整出下一时刻的参数$\phi_{i+1}$。因为我们假设了task的转移是具有markov chain的性质的，所以我们可以将上面的式子写成：
$$
min_{\phi_{1}}\ E_{P(T_0), P(T_{i+1}|T_i))}[\sum_{i=1}^LR_{T_i, T_{i+1}}(\phi_{i})]
$$

这里的$R_{T_i, T_{i+1}}$定义为：
$$
R_{T_i, T_{i+1}}(\phi_i) = E_{\tau_{i, \phi_i}^{1:K} \sim P_{T_i(\tau|\phi_i)}}[E_{\tau_{i+1,\phi} \sim P_{T_{i+1}}(\tau|\phi_{t+1})}[L_{T_{i+1}}(\tau_{i+1, \phi_{i+1}})|\tau_{i, \theta_{i}}^{1:K}, \phi_i]]
$$
与MAML中RL定义的不同的，很直观的是：下一参数的环境不再是相同的$T$，而是下一时刻的$T_{i+1}$。

在实际上，并不是从一个$\phi_0$开始一步步调整到$\phi_L$的，而是每一步都从$\theta$调整成$\phi_n$。因为这可能存在一个问题，就是如果这个L比较大的话，如果是根据上时刻的参数来调整下一参数的话，一方面方差比较大，同时容易导致diverge，所以的实际做法，并没有根据之前的参数$\phi_{t}$来调整，而是在每次调整的时候，从同一参数$\theta$来调整。这里大家可以类比蒙特卡洛的更新，和one-step来想像理解，所以实际上写成：
$$
R_{T_i, T_{i+1}}(\theta) = E_{\tau_{i, \theta}^{1:K} \sim P_{T_i(\tau|\theta)}}[E_{\tau_{i+1,\phi} \sim P_{T_{i+1}}(\tau|\phi)}[L_{T_{i+1}}(\tau_{i+1, \phi})|\tau_{i, \theta}^{1:K}, \theta]]
$$

同样，我们简单地将MAML中的梯度来修改为下一时间的$T_{i+1}$的梯度：
$$
\bigtriangledown_{\theta, \alpha}\ R_{T_i, T_{i+1}}(\theta, \alpha) = E_{\tau_{i, \theta}^{1:K} \sim P_{T_i}(\tau|\theta),\tau_{i+1, \phi}\sim P_{T_{i+1}}(\tau|\phi)}[L_{T_{i+1}}(\tau_{i+1, \phi})[\bigtriangledown_{\theta, \alpha}log \pi_{\phi}(\tau_{i+1, \phi}) + \bigtriangledown_{\theta}\sum_{k=1}^{K}log\pi_{\theta}(\tau_{i, \theta}^k)]]
$$

所以在训练的时候，通过对MAML简单的调整，我们就可以获得相应的伪代码：
![屏幕快照 2018-02-09 下午12.57.21](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Continuous%20Adaptation%20via%20Meta-Learning%20in%20Nonstationary%20and%20Competitive%20Environments/media/1.png)

注意在伪代码中，因为我们是从$\theta$开始调整参数的，所以我们采用$\pi_\theta$在task $T_i$做采样。但是在实际使用时，我们希望在每个task下面，agent都表现出好的效果，比如在task $T_i$必然使用更新好的policy $\pi_{\phi_i}$来与别的agent play，而不会采用$\pi_\theta$（因为$\pi_\theta$只是方便调整，效果不一定好）。

这点需要考虑的主要原因是：实际在nonstationary的环境中，我们并不能访问同一个task多次，虽然在训练的时候，我们可以固定住task，然后用$\pi_\theta$来做采样，然后更新参数。但是在实际使用，或者测试时，我们不可能只用$\pi_\theta$采样，因为当$\pi_\theta$采样完后，面对的又是新的环境，更简单地讲：因为参数都是从$\theta$开始调整的，所以具体操作是需要两个步骤的，一，利用$\theta$来采样，更新，二，用新参数play，但是在实际使用中，我们不可能一直用$\theta$来采样，因为我们的目的是采用一个快速，更好的policy来面对non stationary的环境，所以在使用时，我们应该是采用新的policy，而不是$\pi_\theta$。

用一句来说，就是：为了更好的效果，采用最新调整的policy，但是我们是从\pi_theta开始调整参数的。

更进一步，导致的问题就是：就是我们想要好的policy的效果，所以在实际使用中，比如使用最新的policy $\pi_\phi$来与别的agent play，那么采样出来的轨迹必然与$\pi_\theta$不同，这样的问题，其实就是off-policy的问题，用上importance weight来在实际使用中调整策略：
![屏幕快照 2018-02-09 下午1.06.32](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Continuous%20Adaptation%20via%20Meta-Learning%20in%20Nonstationary%20and%20Competitive%20Environments/media/2.png)
所以采用importance weight correction：
$$
\phi_i = \theta - \alpha \frac{1}{K} \sum_{k=1}^{K}(\frac{\pi_\theta(\tau^k)}{\pi_{\phi_{i-1}}(\tau^k)})\bigtriangledown_{\theta}L(\tau^k)
$$

对于新参数的更新，有时候我们并不会只采用一个step来更新，因为如果梯度较小的话，为了更新的差别大些，我们必然需要设置比较大step size，但是大的step size对于相对大的梯度时，又不能很好的反应整个loss的变化（主要的原因是我们采用采样的方式来计算loss的，类比sgd与batch gd，所以在这里，采用了多次的gd来调整一下时刻的参数，同时也可以将step size当成学习的一个目标来调整。即：
$$  
\phi_i^0 = \theta\\ 
\phi_i^m = \phi_i^{m-1} - \alpha_m\bigtriangledown_{\phi_i^{m-1}}L_{T_i}(\tau_{i, \phi_i^{m-1}}^{1:K}) m = 1,2 ...M-1\\
\phi_{i+1} = \phi_i^{M-1} - \alpha_M\bigtriangledown_{\phi_i^{M-1}}L_{T_i}(\tau_{i, \phi_i^{M-1}}^{1:K})
$$



# 实验环境
这里的实验环境与内容比较多，同时方法和做法已经写的比较多了，所以我在这里略去实验环境，需要的同学自行去看一下论文。

![屏幕快照 2018-02-09 下午1.46.47](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Continuous%20Adaptation%20via%20Meta-Learning%20in%20Nonstationary%20and%20Competitive%20Environments/media/3.png)
![屏幕快照 2018-02-09 下午1.46.53](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Continuous%20Adaptation%20via%20Meta-Learning%20in%20Nonstationary%20and%20Competitive%20Environments/media/4.png)
![屏幕快照 2018-02-09 下午1.47.01](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Continuous%20Adaptation%20via%20Meta-Learning%20in%20Nonstationary%20and%20Competitive%20Environments/media/5.png)

