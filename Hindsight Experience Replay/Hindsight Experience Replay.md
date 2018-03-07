# Hindsight Experience Replay

# 核心问题
1. 研究问题：在稀疏reward的环境中训练policy
2. 假设条件：可以控制想要学习的goal，同时可以直接控制reward function，甚至可以使用goal来修改observations，同时policy可以依据goal来做不同action
3. 主要想法：将goal加入transition tuples $（s，a，r, s', g）$，利用goal之间的相似性，使得完成不同goal的transition tuples可以辅助其他的goal做training
4. 解决方案：使用policy在环境中做交互时，收集transition tuples $（s，a，r, s', g）$，除了当前需要完成的goal，并使用一定的方法（final，future，episode，random）挑选其他的goal $g‘$来生成transition tuples $（s，a，r, s', g'）$并存储在replay buffer中，然后再从replay buffer中抽取数据来training

# 一些局限
这一部分可能看完下面再看比较好理解。但是我觉得是比较有价值的一部分，所以就放在前面。

其实her目前看上去局限很多（当然也就是改进的地方）。比如这里就直接假设reward和goal可以直接控制的，但是很多情况下并不是，可能我们就是要实现固定的几个goal，而且不知道这里面的reward，同样goal之间的关系可能不是特别紧密，那么HER该怎么用呢？（基本假设都出现了问题）

另外就是这里实验的设计，goal变了，其实导致了$（s，a，r, s', g）$中s的改变，但是很多时候，这个s里面是会体现goal的，但是我们无法直接修改s，比如玩video game，里面肯那个有
her的局限，比如这边的假设实在是有点多，特别是goal这一块，前提就是你应该需要了解这个任务，其实就是这边修改了observation-》比如breakout，并不能用（主要强调observatoin对于goal的影响）

写一下her的具体流程

实验
说一下dense的reward设计并不是特别好
另外就是single goal的时候有帮助，注意下observation，那里面其实有个关于goal的绝对位置




# 思路
人类在学习的时候，很多时候并不一定要完全达到自己的特定目标，才会学习到特定的经验和技术。人类在学习的过程中，可能会尝试不同的手段／方法来做一件事，虽然可能这个方法可能在你当前的特定任务$T$上不奏效，但是这样的方法可能完成了一些其他的事情$T'$，当下次你需要做这些其他事情$T‘$时，你就可以使用这次的经验来完成。以我自己的理解来举个例子就是：比如做化学实验进行探索，可能你的方法这次合成了某种化合物A，但是你的目的是B，虽然没有完成合作化合物B的任务，但是这次的方法可以指导你如果合成化合物A。

所以基于上面的思路，如果我们的目的是做一类比较接近的goal的话，或者我们能构造出与当前goal比较接近的一系列goal的话，我们就有可能利用另外的goal来衡量policy在环境中的trajectory$<s_0, a_0, s_1, a_1, ...a_{n-1}, s_n>$的好坏，虽然在大部分的goal中这个轨迹可能是比较差的，但是如果我们的goal是要达到$s_n$的话，那么这个policy其实好的。然后利用这个trajectory来进行学习，这里值得注意一下，我个人的理解是：对于这些goal需要具有一定的联系（内在的相似性），这样这个trajectory训练出来的效果才有可能对于完成另外goal有帮助。

这里我就特意指出来，在论文里面没有提到，但是我觉得很重要的是：对于一些机器手推动物体到达某个位置，这个位置其实就我们的goal，在论文里面的实验环境描述中有提到observation中有一部分信息是：goal（要移动的物体）距离机器手的相对位置的描述。所以虽然每次goal需要推动到的goal的绝对位置不一样，但是这里observation的是相对位置，移动相对位置相同的物体到相对位置相同的goal，其实是可以相同的policy的，所以这部分虽然说是不同的goal，但是内在的本质是非常接近的，所以上面利用trajectory能够有效的加速和训练policy，很大一部分是这样的环境的设置。那么必然，在很多情况下这个goal不好构造，或者observation并不像上面描述的具有一些很巧妙的结构的话，这个方法我觉得运用起来效果应该需要讨论。相反，如果你要解决的问题能具有上面描述的性质，那么你就可以考虑使用HER了。


# Hindsight Experience Replay
理解Hindsight Experience Replay（HER），其实最需要补充的一点就是：Multi-goal RL。Multi-goal RL与普通传统的RL最大的不同就是：显示地知道需要完成多个任务。HER基于Universal Value Function Approximators的思路来设计算法，其实可以简单地理解成，我们在开始一个episode时候，是能知道当前episode想要完成那一个goal的，那么其实就是我们能够拿到的信息不只是state $s$，同时也有需要完成的goal $g$，然后policy再依据goal来选择action：$\pi(a|s, g)$。

其实考虑goal，整个环境的很多部分都发生改变，比如：reward function是需要重写成$r(s, a, g)$；state $s$中包含的信息可能也与goal相关，或者更严谨地说observation是与goal相关的，比如原来是$o(s)$，现在是$o(s, g)$。

所以当我们需要完成multi-goal时，当policy生成一条轨迹trajectory $\tau$时，如果只用一个goal来衡量的话，其实这条$\tau$使用是有点浪费，特别是当我们的goal给的reward比较稀疏时更明显，比如：goal $g_n\ 到达目标s_g\ : if\ s_t == s_g: reward = 1, else\ reward = -1$。如果这条轨迹恰好到达了$s_g$，那么还好，贡献了一个reward为1的transition tuples $（s_{n-1}，a_{n-1}，1, s_g, g_n）$，其他的transition tuples与goal $g_n$相结合，那么reward都是-1（如果没有到达$s_g$的话，那么reward都是-1，对学习没有任何帮助），并不能有效地帮助$g_n$训练，也不能帮助别的goal训练。

所以既然感觉有点浪费，就会想要利用起来，这部分也就是HER做的事情：如果我们能够知道r(s, a, g)的话，那么对于上面采样出来的$\tau$中的$(s, a, r(a, s, g), s', g)$，我们可以选择不同的goal，让这里面的reward变成1，就是意味着：这个transition tuples能够有效地帮助这个goal进行学习。那么replay buffer中reward为1的transition tuples数目就得到了一定的提升，可能就能够有效地帮助agent学习。

下面是HER的算法，简单地解释一下就是：利用当前policy在环境中交互获得trajectory $\tau$，然后将$(s, a, r(a, s, g), s', g)$存储在replay buffer中，然后再挑选一些其他的goal对这个trajectory $\tau$中的g和r做修改，然后存储在replay buffer中，之后就是普通的基于replay buffer算法中常见的从buffer中sample，然后训练等过程中。
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Hindsight%20Experience%20Replay/media/1.png)

那么关于如果挑选其他的goal就是一项很玄学的地方了，在论文里面提出了几种不同的方法：
* final — goal corresponding to the final state in each episode
* future — replay with k random states which come from the same episode as the transition being replayed and were observed after it
* episode — replay with k random states coming from the same episode as the transition being replayed
* random — replay with k random states encountered so far in the whole training procedure


# 实验
## 实验环境
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Hindsight%20Experience%20Replay/media/2.png)
这里有三种任务：

1. Pushing. 把物体推到指定的位置2. Sliding. 推动物体，使它滑动到某个位置3. Pick-and-place. 拿起物体，移动到空中的某个位置

在这个环境中：
* reward：在没有到到达goal时，都是-1，到达goal时候为0
* goal：为在空间中随机生成的位置（所以我感觉这也是有效的一点）
* Observations：gripper（机器手）在这个空间中的绝对位置，需要推动物体object和goal相对gripper的相对位置

## Does HER improve performance?
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Hindsight%20Experience%20Replay/media/3.png)
比较一下，有没有HER对于不同任务的学习效果，同时加入一些探索的方法做比对，看看是不是探索太少了。结论就是：HER能够有效的学习。

## Does HER improve performance even if there is only one goal we care about?
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Hindsight%20Experience%20Replay/media/4.png)
其实就是固定每个episode要完成的goal相同goal，而不是在空间中随机生成goal。这里的实验效果是有效的，可能看上去很奇怪，但是如果你有仔细观察Observations中，goal等是用相对位置表示的，所以虽然用HER生成的goal不同，但是如果相对位置表示相同时，所需要做的action是相同的，所以是能够有帮助学习的（实验环境的设置关系很大！！！！）

## How does HER interact with reward shaping?
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Hindsight%20Experience%20Replay/media/5.png)
这部分，我感觉就是真没有好好调reward function，但是毕竟别人想要体现的是稀疏reward

## How many goals should we replay each trajectory with and how to choose them?
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Hindsight%20Experience%20Replay/media/6.png)
这里就是上面提到生成goal中的k的选择

## Deployment on a physical robot
![](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Hindsight%20Experience%20Replay/media/7.png)
测试了一下，直接部署在真正的robot上，Initially the policy succeeded in 2 out of 5 trials. It was not robust to small errors in the box position estimation because it was trained on perfect state coming from the simulation. After retraining the policy with gaussian noise (std=1cm) added to observations8 the success rate increased to 5/5. 

