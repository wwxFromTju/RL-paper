# Mix & Match – Agent Curricula for Reinforcement Learning

# 核心问题
1. 研究问题：如何在一些复杂问题中加速复杂policy训练
2. 假设条件：简单policy在复杂环境可以训练出一定的效果
3. 主要想法：先训练一个简单粗糙的policy，然后用来加速复杂policy的训练
4. 解决方案：采用类似mixture expert的方法，加权简单policy与复杂policy来作为交互的policy，在训练时动态调整权重，在前期利用简单policy的学习速度与探索效率，后期集中在复杂policy上，使得收敛的policy效果更好。

# 动机
深度强化学习解决了越来越多的问题，所以人们也开始通过设计越来越难的环境与任务，探索深度强化学习解决更复杂问题的能力。为了在更复杂环境中获得更好的效果，我们通常会选择更复杂的policy（如果用NN来表示policy，那么就是更多层，每层的size更宽，考虑LSTM之类），但是选择复杂的policy也带来学习速度更慢的后果，即可以理解为学习的曲线更平缓。这是一个不可避免的问题：更深的网络，bp回去的效果越差，同时每次bp完，policy的变化也不会变化太大，都是一个缓慢调整的过程。虽然有很多并行的框架可以加速policy的训练，但是没有实际解决训练慢的本质问题。

一个做法是利用：课程学习通过逐步提升任务／环境的复杂度／难度，利用这个过程来逐步训练，获得一个agent，或者利用这些不同任务／环境上的agent来蒸馏出一个通用的agent。这样做法的原因：是利用更容易环境／任务的reward反馈更快，需要policy的精细程度更粗糙，来加速训练过程。
但是这样的做法有个前提：
1. 能够修改环境
2. 需要对环境有很深刻的理解，才能够训练出一系列难度递增的任务

虽然存在一些自动生成课程的paper，但是实际效果都蛮一般的。这篇文章并没有利用这种修改任务／环境的做法，而是从agent内部policy的表示（复杂度）出发，通过调整实际使用policy的复杂度，在前期使用简单的policy，加速agent的训练，在后期使用复杂的policy，获得一个非常好的效果。所以这里的一个非常重要的假设就是：简单的policy能够在这个环境上也能学出一定的效果，这样才能起到前期加速的效果。如果这个困难的任务本身就需要复杂的操作才能完成，那么估计用简单的policy学不出来，那么加速的效果也就不用想了。

 <!--
Net2Net不改变网络表征，通过supersets/subsets 别的网络和identity mapping，逐步增加网络容量capacity。一个follow的工作扩展了这，类似于结构上的课程学习，先训练小的网络，然后训练大的。

knowledge transfer／distillation：从别的模型迁移能力（策略）过来，而不需要考虑模型结构。开始在model compression上，在多个distinct policies into single one


实际用的策略是一堆策略的mixture，最终我们的目的是训练最后一个策略。

通过这个kl散度的loss，来让policy之间的distribution比较接近，即互相分享了学到的knowledge。

同时证明了这个是mm与pi2的kl散度的upper bound，所以降低这个loss同时也是使mm最终收敛到pi2（巧妙！）

后面比较普通，说明pi2能够向pi1学习
-->

# 框架
![屏幕快照 2018-07-07 下午8.38.14](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Mix%20%26%20Match%20–%20Agent%20Curricula%20for%20Reinforcement%20Learning/media/1.png)

如图所示，这里的框架分成两个部分：
1. 不同的底层的policy: $\pi_1, \pi_2 ... \pi_k$
2. Mixing policy: $\pi_{mm} = \sum_{i} w_i * \pi_i$

其中$\pi_1, \pi_2 ... \pi_k$的训练难度／复杂度依次增加。

### mixing
目的是最终学习出一个比较好的policy $\pi_k$, 即最终为$\pi_{mm} = \pi_k$。在获得最终policy的过程中，我们可以通过设置不同权重 $\w_i$ 来获得不同复杂的mixing policy $\pi_{mm}$。所以在直观上，可以在开始的时候设置$w_1 = 1$，其他的的等于0，然后随着训练时间的进行，依次增加后面policy的权重，减少前面policy的权重。这个过程可以简单地直接增加，但是实际上，我们希望的效果是：通过控制这个权重，来调节学习中的dynamics，同时最大化最终策略的效果。（而不是立即的回报）

但是只是做简单的mixture policy的话，那么实际上这些policy不会share学到的知识，所以简单policy学习出的knowledge并没有用来加速那些复杂policy的学习过程，而这样做mixing反而会损害一起做mixing的policy效果，举个例子：
一个policy通过梯度反传学习到了有效的policy，但是这一部分knowledge并不会在计算loss的时候传递到其他的policy上，同时可能因为其他policy几乎无效，所以当前bp之后可能就变成这个学习到的policy的权重为1，而其他的为0了，这样才是一个最优的学习过程。

对于mixing的部分，可以从两个方面来看：一个是通过categorical random variable来随机挑选不同策略，另外一个就是直接mixture这些策略。从expected gradients的角度来说这两种做法是等价，但是第一个种实际上需要多次抽取，然后交互，计算多个gradients的均值才等价于第二个，如果只是进行抽取几次，那么实际上这个方差会蛮大的。所以采用第二种。

$$
E_{i \sim c}[\bigtriangledown_{\theta}\pi_i(a|s, \theta)] = \sum_i w_i\bigtriangledown_{\theta}\pi_i(a|s, \theta) = \\
\bigtriangledown_{\theta}\sum_i w_i \pi_i(a|s, \theta) = \bigtriangledown_{\theta}\pi_{mm}(a|s, \theta)
$$


### knowledge share
所谓share knowledge，其实就是在相同state下采用相似的action。那么我们就可以采用限制一系列不同的policy，在相同state下的action distribution比较接近即可，具体的可以定义一个不同policy间action distribution区别的loss：
$$
L_{mm} = \sum_{i,j}D(\pi_i, \pi_j, i, j, w_i)
$$
其中$D$代表action distribution的不同程度，所以通过reduce这个loss，那么不同policy在相同的state下，就会具有相似的action distribution，那么就share knowledge了。在具体训练的时候，我们可以类似unsupervised auxiliary tasks的做法，将这个loss加在训练policy的loss中即可。

更为具体的，假设我们采用两个策略（实际上这篇文章都采用两个策略，哈哈哈哈，发现大家在开始定义的时候都喜欢往大的地方定义），那么我们可以定义loss为KL散度：
$$
L_{mm} = \frac{1-\alpha}{|S|} \sum_{s \in S}\sum_{t}D_{KL}(\pi_1(\bullet|s_t)|| \pi_2(\bullet|s_t))
$$
而且这个loss有个更有意思的地方，如果我们考虑下面这个KL散度：
$$
\hat{L_{mm}} =D_{KL}(\pi_{mm}(\bullet|s)|| \pi_2(\bullet|s)) \\
= H(\pi_{mm}(\bullet|s)|| \pi_2(\bullet|s)) - H(\pi_{mm}(\bullet|s)) \\
= H((1-\alpha)\pi_1(\bullet|s) + \alpha\pi_2(\bullet|s) || \pi_2(\bullet|s)) - H(\pi_{mm}(\bullet|s)) \\
= (1-\alpha)H(\pi_1(\bullet|s)||\pi_2(\bullet|s)) - (H(\pi_{mm}(\bullet|s)) - \alpha H(\pi_2(\bullet|s)))
$$
从上到小的步骤分别为：把KL写成熵，然后把$\pi_{mm} = (1-\alpha)\pi_1 + \alpha\pi_2$带入（因为就只两个policy，所以下面就用$\alpha$代表最终policy $\pi_2$的权重），然后拆开即可。
同时，我们将原来的loss简单写成:
$$
L_{mm} = (1-\alpha)D_{KL}(\pi_1(\bullet|s)|| \pi_2(\bullet|s))\\
= (1-\alpha)H(\pi_1(\bullet|s)||\pi_2(\bullet|s)) - (1-\alpha)H(\pi_1{\bullet|s})
$$
所以上面两个相减：
$$
L_{mm} - \hat{L_{mm}} = \\
= H(\pi_{mm}(\bullet|s)) - （\alpha H(\pi_2(\bullet|s)))
 + (1-\alpha)H(\pi_1{\bullet|s})
$$
因为entropy是concave的，所以加权和必落在下面，即$L_{mm} \geq \hat{L_{mm}}$，所以实际用的$L_{mm}$是$\hat{L_{mm}$的upper bound，所以去reduce $L_{mm}$也就push $\pi_{mm}$与$\pi_2$相互接近，所以最终理论上可以让$\pi_{mm}$收敛到$\pi_2$，也就是咱们实际要的policy！！！！！！！cool！！！！！！！

### Adjusting $\alpha$ through training
虽然上面降低$L_{mm}$可以让$\alpha$向1移动，但是实际该怎么移动还是不清楚。当然我们可以手工构造alpha的变化过程，但是相同的变化过程，产生的训练效果依赖于不同的问题。虽然可以很简单递增$\alpha$，同时能够光滑地移动$\alpha$，来让RL learning stable。但是手工的设计的做法，可能会丧失一些动态调整alpha来获得最优策略的结果。

同时因为关注的是alpha的变化过程（而不是一个单一的定值），所以这个空间特别的大，而且不好调，通常的grid search／bayesian optimisation不太可以直接用过来，而且估计效果不好。这里采用Population based training（PBT）来做，通过online的一堆agent，然后看他们参数的效果，来动态的调整一个群体的alpha，这样就能够获得一个alpha的变化过程。通过判断不同population的性能，比如在mixed policy间switch，那么可以对于这些population都eval k个episode的性能，然后对比即可。在explore的话，可以在好的alpha的基础上，随机增／减一个数值，约束在合理的范围内。这样的PBT实际上就将alpha的动态调整过程构造出来的了。效果类似于进化算法，有兴趣可以自己去看PBT的论文

# 实验与分析
这边在三个地方验证了这个想法的有效性，环境都为类似雷神之锤的游戏，即deepmind lab的环境中：
1. 逐步提升action space的大小
2. 逐步变难网络结构
3. 在多个任务下测试

具体环境类似于：
![屏幕快照 2018-07-07 下午9.32.00](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Mix%20%26%20Match%20–%20Agent%20Curricula%20for%20Reinforcement%20Learning/media/2.png)


三个地方分别采用的网络结构：
![屏幕快照 2018-07-07 下午9.32.56](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Mix%20%26%20Match%20–%20Agent%20Curricula%20for%20Reinforcement%20Learning/media/3.png)

### action space
游戏里面是6维的action space，因为有的维度是旋转角度，控制比较灵活，所以最终的action spce非常大：4 x 10 ^ 13。之前的大部分做法就是先人工reduced掉这个空间的大小，在一部分的subset action space上面做这个问题。subset带来的好处是：空间小了，探索效率自然就高（引入人为的偏置）。但是因为是subset，所以可能影响最终收敛policy的效果。

所以在这里的两个策略为：$pi_1$为9个动作（small action space），$pi_2$为756个动作（限制x，y的变化，即big action space），然后采用factorised policy + conditional independent。在action space这个设置中，pi间共享底层的参数(见结构图）。

从实验结果来看：small action space的policy学习的比较快，但是最终收敛的performance比big space的policy差。M&M的速度前期速度和small space的差不多，同时最终的结果比big space的好。这边给的假设是：前期的small action space的policy带来对于环境比较好的全局探索。
![屏幕快照 2018-07-07 下午9.35.21](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Mix%20%26%20Match%20–%20Agent%20Curricula%20for%20Reinforcement%20Learning/media/4.png)


然后实验进一步分析M&M的效果，对比了M&M的两个扩展：
1. shared head，共享最后输出层的参数，然后对于小的action space直接对于输出，采用mask掉超出空间的输出，然后对于剩下的做renormalising。
2. masked KL，因为$pi_1$与$pi_2$的action space不一样，所以算KL的时候只在双方重合的action space上算

这两个新的方法和原来的差不多，或者略差。masked kl可以让agent比较free地去探索，学习没有$pi_1$中的action，所以最终性能有时候会略好。
![屏幕快照 2018-07-07 下午9.36.37](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Mix%20%26%20Match%20–%20Agent%20Curricula%20for%20Reinforcement%20Learning/media/5.png)


另外看一下这个试验的alpha变化（所以实验都init为0，然后不强制$\alpha$的变化趋势）与action使用变化。alpha比较快速地变到1，所以代表小的action space为加速前期的学习过程，但是在后期没有使用。对于action使用的分布，同样可以看出后期采用复杂的policy和control
![屏幕快照 2018-07-07 下午9.40.47](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Mix%20%26%20Match%20–%20Agent%20Curricula%20for%20Reinforcement%20Learning/media/6.png)

### agent architecture
在结构设计中，和action space的设置类似，但是第二层的lstm在简单结构中被替换成全连接，然后共享底层cnn与output层的权重，同时采用540的action space。具体看上面的结构图。

换成全连接的为reactive policy（写成FF），即反应式的policy，对于一个state给予一个固定的action distribution，虽然能够比较直观地解释每个state下面action distribution的目的，但是可能存在一些无效的探索，因为不知道之前探索过什么（比如假设输出为确定性的action，那么设置可以陷入死循环），而具有lstm的policy相对而言可以缓解上面的问题，提升最终的performance。

相比较FF，LSTM与M&M的最终效果比较好（FF，LSTM代表只用对应的部分结构），M&M的效果最好。但是从速度而言，M&M反而是最慢的。
![屏幕快照 2018-07-07 下午9.42.37](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Mix%20%26%20Match%20–%20Agent%20Curricula%20for%20Reinforcement%20Learning/media/7.png)


为了研究到底是哪里慢了，对比了采用skip connection的结构，将FF与LSTM的输出加起来，然后再激活的结构（类似与M&M，但是没有alpha的变化，比较像aplha=0.5），M&M的学习动态和这个比较像，但是M&M还是慢，所以这面下个结论是M&M在这丧失了开始加速的效果。同时想要了解M&M带来的提升到底是core的type不同导致，还是类似KL的loss带来的。这里就对比了一下M&M的结构，但是两个core都是lstm。通过实验可以发现，效果又差又慢，所以这个提升是kl的loss带来（同时加上切换）
![屏幕快照 2018-07-07 下午9.44.53](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Mix%20%26%20Match%20–%20Agent%20Curricula%20for%20Reinforcement%20Learning/media/8.png)

然后看一下alpha的变化，最终还是都到1，这个过程中存在一些有意思的点：
1. Lasertag level的alpha变化比较慢，结论为环境比较难，需要比较复杂的policy，所以$\pi_2$训练的比较慢，所以switch就比较慢
2. Nav 01的alpha先到1，然后到0.5，然后再到1，感觉就是训练时候$\pi_2$变差，然后回退，之后通过训练$\pi_2$，再switch到1
![屏幕快照 2018-07-07 下午9.45.07](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Mix%20%26%20Match%20–%20Agent%20Curricula%20for%20Reinforcement%20Learning/media/9.png)

### multitask
最终的目的是：学习一个单一的任务在multitask上有比较好的效果。

原来的做法是：同时在多个任务上训练，共享模型参数。这样带来的两个问题：
1. 容易被环境reward数值相对大的环境所影响，最终policy的效果在这类环境上比较好，在其他的环境上效果差（不同环境间reward尺度不同）
2. 容易被更快产生sample的环境影响，直观上就是这个环境学的更快，导致其他的环境相当于在这个环境的best model的一定范围内做微调

三个环节：一个是explore object location small的环境，有数值比较高的reward，而且比较容易学。另外两个是laser tag的不同level，比较难学。

原始的multitask的训练方式的确暴露了相应的问题：在reward数值高的，容易学的env上面效果，在其他的效果相对差。然后M&M采用每个环境一个特定的$\pi_1$，然后多个环境共享一个multitask的$\pi_{mt}$，然后在每个环境里再做mixture。我们真正要学习的是$\pi_{mt}$，所以mixture只是在学习中调整和sample出轨迹（因为这边eval采用的alpha为1，所以$\pi_{mt}$效果好很直观）。而且在多个task上面效果好，

![屏幕快照 2018-07-07 下午9.45.37](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Mix%20%26%20Match%20–%20Agent%20Curricula%20for%20Reinforcement%20Learning/media/10.png)


