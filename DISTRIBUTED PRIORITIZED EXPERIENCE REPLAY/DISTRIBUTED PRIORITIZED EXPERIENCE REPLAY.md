# DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY

# 核心问题
1. 研究问题：如何更有效地利用计算资源来加速DRL的训练速度
2. 假设条件：数据生成成本低，可以同时大规模地获得的policy生成的数据
3. 主要想法：利用大量不同的policy来探索环境，获得对于环境更全面的认识，以此来‘over fitting’一个policy出来（这里的over fitting是指对于这个环境，而不是数据）
4. 解决方案：利用off-policy的做法，采用一个learner，多个actor。actor交互获得数据，放在learner的中心buffer中，然后一段时间从learner处更新policy参数

# 动机
当前深度学习提升的趋势为：
1. 利用更多的计算力，比如多个GPU卡同时训练来更快地计算梯度，然后更新model
2. 根据问题来设计更有效的model，比如resnet里面的残差block让梯度BP更有效
3. 更大的数据，比如从MNIST到ImageNet，因为更多的数据，就可以刻画更为一般的特征与性质，比如MNIST是白天鹅，那么你的NN学完之后就都认为天鹅都是白的，但是ImageNet中存在一堆不同颜色的天鹅，那么通过ImageNet学完后，NN就会认为天鹅的外形比较重要，颜色有很多。同时更多的数据，就可以用更大的网络，而不用担心over fitting

所以如果我们借鉴DL的趋势，一个很直观的做法就是利用更多的计算力，比如A3C：异步的参数更新与数据生成，只用cpu，thread尺度，只用single machine。GA3C：在A3C的基础上，更合理地利用GPU资源。PAAC：更有效地利用GPUs的资源等等，同时存在一些更通用的RL训练框架，比如Gorila之类的，提供了参数服务器等组件，类似DL的方法来加速等等。

但是当前很多加速的方法还是集中在计算力上，这里说的计算力是指说：通过并行加速计算当前policy在环境中不同轨迹的梯度，然后reduce这些梯度来稳定整个sgd的过程，让agent学习得更稳定些，更快些。本质上是利用多个env的副本，同时evaluate当前这个policy，然后来做update，所以利用的是：共享梯度。

但是共享梯度存在一个比较敏感的点：这个梯度与当前的weights相关性特别大，weights稍微一改变，那么同一条轨迹（数据）对于weights产生的梯度就会发生比较大的变化，即可以理解为梯度对于policy的更新非常敏感。所以在共享梯度的设置下，虽然你可以用off-policy的方法（比如Q-learning，DQN），但是因为梯度的敏感性问题，本质上的data efficiency还是非常低的。

所以一个很直观的想法：在这种并行的情况下，既然可以用off-policy的方法，那么我该如何提升data efficiency呢？既然DQN中使用buffer将data存起来，然后通过不断地从这个buffer中抽取data来提升自己的data efficiency，同时提供一定的iid的性质，那么我们就可以很直观从共享梯度，变成共享data，放在一个buffer中（绕了一圈，又要用buffer，A3C当初说自己的贡献就是提到说可以不用buffer，减少内存的占用，而且不用sample，速度更快）。

如果只是共享data的话，那么本质上就是对于buffer更快速地更新，buffer中的数据能够更准确地反应online NN的policy。

但是如果我们换一个角度来考虑问题，从DL提升第三点来看问题（即更大的数据）：我们是否能够构造出一个更能放映出env信息的数据集？比如：我们能够获得环境中的所有信息，然后利用这些信息来构造出一个数据集，那么其实我们就不用仿真env，直接对于这个数据集来做学习就好了。

构造出一个更能反映出环境信息的buffer，一个很直观的做法就是利用更多样的behavior policy来生成数据，一种做法就是控制$\epsilon-greedy$中的$\epsilon$为不同的值。但是其实这样做也有一些弊端，就是behavior policy产生的数据分布与learner的差别大，会影响最终的学习效果，类似于探索-利用的困境，实际论文证明了还是利大于弊。

从利用NN做函数近似的角度，多样性的数据可以避免NN对于当前的policy进行over fitting。

# 框架
![屏幕快照 2018-06-08 下午4.03.29](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/DISTRIBUTED%20PRIORITIZED%20EXPERIENCE%20REPLAY/media/1.png)
采用一个learner，多个actor的结构：actor中为不同的behavior policy，与环境交互获得数据，然后利用当前的policy计算这个数据的初始优先级，然后将数据存共享buffer中，一段时间从learner处获得最新的policy的weight。learner在共享buffer中采样，更新weights，同行死调整buffer里面数据的优先级。

## 算法伪代码
![屏幕快照 2018-06-08 下午4.07.28](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/DISTRIBUTED%20PRIORITIZED%20EXPERIENCE%20REPLAY/media/2.png)
![屏幕快照 2018-06-08 下午4.07.48](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/DISTRIBUTED%20PRIORITIZED%20EXPERIENCE%20REPLAY/media/3.png)
这里需要注意的是：
* actor先将数据存在local buffer中，然后一段时间后再放到learner的buffer中，结合后面对于数据的处理（图片信息在actor做压缩，然后learner处再解压），应该是考虑到通讯的带宽
* 对于buffer的大小为soft-limited，不限制actor往里面放，learner每隔N次更新后，扔掉旧数据，保持buffer大小的limited

## Ape-X DQN／DDPG
对于DQN的部分，采用了Double Q-learning，dueling network architecture
multi-step bootstrap targets:
$$
G_t = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n - 1}R_{t+n} + \gamma^n q(S_{t+n}, argmax_{a}q(S_{t+n}, a, \theta), \theta^{-})
$$
NN的loss为与multi-step bootstrap targets的MSE。

ape-x ddpg的形式和ape-x dqn差不多，就是改成了ddpg的形式，然后critic的更新target变为：multi-step bootstrap targets

# 实验
![屏幕快照 2018-06-08 下午4.20.50](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/DISTRIBUTED%20PRIORITIZED%20EXPERIENCE%20REPLAY/media/4.png)
![屏幕快照 2018-06-08 下午4.21.42](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/DISTRIBUTED%20PRIORITIZED%20EXPERIENCE%20REPLAY/media/5.png)
![屏幕快照 2018-06-08 下午4.21.48](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/DISTRIBUTED%20PRIORITIZED%20EXPERIENCE%20REPLAY/media/6.png)


这里做实验就改actor的数目，然后其他全不变，发现增加actor数目效果不错（因为actor实际上就只影响buffer的更新，其实对于learner是不影响的，所以超参数等不调整是很正常的）。所以一种直观上的做法是在小规模上先调参数，然后直接上大数据，然后再调

这里同时对于之前rl学不好做了一定的假设：探索不有效，到了局部最优，然后ape-x能够更有效的探索

更大的buffer有提升，但是效果不明显，主要的原因是：learning时候关注的high priority experiences。

总结：更好地探索环境和更好地避免过度拟合

