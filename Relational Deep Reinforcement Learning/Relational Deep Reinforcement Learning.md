# Relational Deep Reinforcement Learning

# 核心问题
1. 研究问题：如何在DRL中引入relational，abstract来提升DRL效果，处理复杂任务，做planning
2. 假设条件：可以将环境编码成一个二维的输入，每个区域代表一个entity
3. 主要想法：将relational的关系转化成attention，利用attention来代表两个entity的关系。隐式地将relational引入NN结构中
4. 解决方案：将问题编码成一个二维输入，然后对于输入做变换，成为N个entity，对于这N个entity采用multi-head dot-product attention (MHDPA)来表示entity间可能存在的relational，希望通过这个步骤，NN能学出对于实际问题输入中的entity概念，同时能够学习出entity间的relational


# 动机
当前DRL在很多任务上都取得很好的效果，通常认为一个原因是：通过NN来learn出一个很好的internal representations来进行policy的表示。但是很多时候训练DRL需要比较高的成本，比如由于data efficiency的问题，需要比较大量的数据。此外DRL的policy体现出对于环境的overfit的倾向，比如说对于env稍微的修改，整个policy效果衰减严重，即可以理解为policy可能并没有学习出一个对于要解决的问题／env的比较好的抽象刻画。

既然觉得当前DRL存在上述等一些列问题，那么是否可以通过让agent能够做更高维度的抽象，和学习这些抽象的relational，来加速DRL的学习过程，同时让policy具有更好泛化性呢？直观上，逻辑推理能够比较好的处理抽象entity的关系，其中RL与逻辑推理的结合：Relational RL（RRL）结合了逻辑推理与RL的能力，既然DRL没有抽象的能力，那么能不能利用RRL来提升DRL的效果？所以我们可以借用了RRL的insights，希望能够借鉴RRL中inductive logic programming的效果，学习与重用entity与关系来隐式低做reason和relational     representation。

RRL在学习的早期学习出一些实体的概念，比如goal，同时明确state，action的影响，相当于学习这个环境的表示或者信息，然后利用logical facts／rules relevant来加速学习／解决问题。同时在这个过程中inductive bias能够限定policy的需求过程，将policy限定在一个好的范围内，来让我们获得一个比较好的policy的表示。所以如果将inductive biases与DRL相结合，通过操作一系列的entity，那么理论上DRL也可以获得相应的relational能力。即通过引入inductive biases的结构，希望能够learning／computing relations的能力。


# 思路
假设我们能够把一个entity集合表示成一个二维的形式（类似矩阵），那么如果我们直接对于这个矩阵做一次卷积计算，可以理解成通过这个卷积来计算相邻entity（即在二维表示上相近的entity，矩阵中相邻的元素）的relational（卷积代表了relational的计算方法）。实际上这种形式的relational存在一系列conv的NN中，但是这种形式的relational只考虑了空间上相近的entity，没有考虑远的entity。同时在抽象思考时，我们更多是考虑做一件事中entity中前后关系，而不是空间中的关系。所以本质上我们应该计算entity相互之间relational，然后再利用这个信息来帮助做推理等等。

对于这样形式的relational，我们可以将其写成常见的attention的形式。假设已经有计算的entity set，在这个set中可以使用multi-head dot-product attention (MHDPA)来计算interaction。如图所示：把state vector $e_i$分解成query $q_i$，key $k_i$， value vector $v_i$。然后对于query用全部的key来计算unnormalized saliencies $s_i$，然后通过softmax转化成权重$w_i$，然后通过$a_i = \sum_{j=1:N}w_{i,j}v_j$作为attention。写成矩阵形式即为:
$$
A = softmax(\frac{QK^T}{\sqrt{d}})V
$$
其中d为dimensionality of the key vectors used as a scaling factor。


如果我们能够假设entity是二维的一块区域，所以就是利用这块区别去发现别的key区域和高度交互的区域。如图所示：
![屏幕快照 2018-06-15 下午9.12.27](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Relational%20Deep%20Reinforcement%20Learning/media/1.png)


这样的假设能够比较直接地使用在图片这类非结构的输入中，并没有对于内容做特定的假设，所以如果别的数据也可以写成一个类似图片等表示，那么直观上可以直接使用。我们将上述attention部分称为attention block，所以采用了一个attention block来计算non-local pairwise relational。如果采用多个blocks（shared或者unshared）相当于计算higher order relations。采用non-local computations与相同的function来在entity空间中计算entity间的关系，相比只考虑local关系，比如conv，那么能够获得更多关于entity的关系。这部分的计算可以很容易的并行化。


# Box-World
![屏幕快照 2018-06-15 下午9.16.49](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Relational%20Deep%20Reinforcement%20Learning/media/2.png)

环境中，暗灰色的方格是agent，agent可以在这个世界中上，下，左，右走。 单独存在的方格（即除了agent，上下左右均没有其他的方格的方格）是agent可以捡起来钥匙，通过这个钥匙可以开启一次同种颜色盒子，获得其中的钥匙（即为两个相邻方格表示，右边为需要的钥匙颜色，左边为开启这个箱子后可以获得钥匙的颜色），最终的目的是为了获钻石（白色方块表示）

这个问题中，钥匙只能用一次，同时钥匙，箱子，agent为随机出现。所以agent必须学习出一条能够通往钻石的开箱子的顺序，同时必须要能够将pixel抽象成一个个entity来帮助分析。对于env难度刻画为：到钻石的过程中需要开几个箱子，与这个过程中获得钥匙后有几个备选可以打开的箱子（一开错可能就不能通往最终钻石了）。实际training时设置：需要开箱子的数目可能为：1到4，分叉可能为：0到4。
　　
虽然这个环境小：12x12的pixel room，但是由于游戏设置的复杂性，所以完成任务需要对于env中entity reasoning和planning。


![屏幕快照 2018-06-15 下午9.17.05](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Relational%20Deep%20Reinforcement%20Learning/media/3.png)

我们在上面具体的两个环境中做实验，发现用了relational都比baseline好，同时更复杂的环境，需要更多次对于relatiolal的迭代，即更多的attention block有更好的效果（4的效果大于2的效果）。

training用了IMPALA，100个actor收集数据，1个learn对这些数据进行学习（learner学习pi与v）

网络结构为：12个2x2的conv kernel，stride为1，relu；24个2x2的conv kernel，stride为1，relu；然后额外拼接上两个维度：分别是这个cell的x，y（x，y reduce 到 $[-1, 1]$ ）；接variable number个relational module；接feature-wise max-pooling across space，把 k 个 feature reduce 到 k 维向量；然后接一个小的 MLP 到 policy logits 和 $V$

此外，为了能够了解具体上每一个row在observation space中关注那一部分。因为我们加了坐标，所以可以通过attention来观测。这边就只观察了在通往钻石路上的每个entity对应的row的attention，其中一个head关注的是这个锁卡能够开启的下一个锁，另外一个head关注的是agent与这个entity的pair。如图所示：
![屏幕快照 2018-06-15 下午9.17.11](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Relational%20Deep%20Reinforcement%20Learning/media/4.png)


进一步为了避免grb空间的影响，这边替换成one-hot来表示。实际上结论差不多：key关注能够开的箱子，箱子关注能够开他的钥匙，所有entity都关注agent／gem


此外，这边是为了检测是否是真的学习到了relational的关系与abstract。如果真的学到了，那么可以面对新颜色的钥匙和箱子，应该能够做推断来开箱子，同时也可以做更长的规划。
![屏幕快照 2018-06-15 下午9.17.16](https://raw.githubusercontent.com/wwxFromTju/RL-paper/master/Relational%20Deep%20Reinforcement%20Learning/media/5.png)

