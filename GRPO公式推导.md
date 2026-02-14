<h1 align="center">RL基础知识 & GRPO公式推导</h1>

# RL基础概念：S、A、R、π、环境

## 状态s、动作a、奖励r、策略π

在用强化学习训练LLM的语境中，有两种建模级别。
- token级建模：每个生成的token视为一个action（对应PPO）
- sequence级建模：每个完整生成序列视为一个action（对应GRPO）

相应的，把强化学习的概念对应到LLM上：
**token级建模**
- 状态s：
  - 初始状态 $s_1$ ：prompt $x$
  - t时刻状态 $s_t$ ：prompt $x$ + response的前t个token $y_{1:t}$
  - （此处 $y_i$ 指的是序列 y 的第 i 个token）
- 动作a：
  - t时刻的动作 $a_t$ ：response的第t个token $y_{t}$
- 奖励r：
  - 最后一个token：完整序列的奖励（例如答案是否正确）
  - 前面的token：奖励为0
- 策略π：
  - LLM生成第t个token的概率 $P(y_t | x, y_{1:t-1})$

**sequence级建模（只有一问一答）**
- 状态s：只有初始、结束两个状态
  - 初始状态 $s_1$ ：prompt $x$
  - 结束状态 $s_2$ ：prompt $x$ + response $y$
- 动作a：只有一步动作
  - LLM生成的response $y$
- 奖励r：
  - 完整序列的奖励（例如答案是否正确）
- 策略π：
  - LLM生成response y的概率 $P(y | x)$

**sequence级建模（多轮问答）**：常见于多轮agent的RL训练，但也可以全拆成单轮
- 状态s：
  - 初始状态 $s_1$ ：初始prompt $x_{1}$
  - t时刻状态 $s_t$ ：初始prompt $x_{1}$ + 第1轮response $y_{1}$ + ... + 第t轮prompt $x_{t}$ + 第t轮response $y_{t}$
  - （此处的 $x_i,y_i$ 指代的是第 i 个 x,y 序列，而不是序列中第i个token）
- 动作a：
  - t时刻的动作 $a_t$ ：第t轮response $y_{t}$
- 奖励r：
  - t时刻的奖励 $r_t$ ：第t轮回复的奖励（有些方法有中间奖励），如果t是最后一轮则对应终局奖励
- 策略π：
  - LLM生成第t个response的概率 $P(y_t | x_1, y_1, ..., x_{t-1}, y_{t-1}, y_t)$
 
## 环境：确定性环境 & 概率环境

环境描述了给定状态s，执行动作a时，会转移到什么新状态s'，并且会获得多少奖励r。
- 有时s'和r都是确定的，则环境是确定性环境
- 有时s'和r是有随机性的，则环境是概率环境，此时需要用概率分布 $P(s',r|s,a)$ 来描述

具体而言又可以分为s'有随机性、r有随机性等多种不同情况。以下以“步枪打靶”为例说明。

> 步枪打靶游戏：让步枪瞄准某个坐标并开枪，打中则获奖。只有一步决策，初始/结束两个状态
> - 初始状态s：步枪当前瞄准坐标 $(x_1,y_1)$
> - 动作a：“令步枪瞄准新坐标 $(x_2,y_2)$ 并开枪”的指令（把移动和开枪视为一整个动作）
> - 结束状态s'：步枪实际瞄准的新坐标 $(x'_2,y'_2)$ 
> - 奖励r：打中 $(x_2,y_2)$ 位置的靶则奖励1，否则为0
>
> 随机性的来源：
> - s'的随机性：动作执行可能不准确，命令枪瞄准 $(x_2,y_2)$ 和枪实际瞄准 $(x'_2,y'_2)$ 可能并不相同
> - r的随机性：动作执行产生的效果可能不稳定，枪就算能瞄准 $(x_2,y_2)$ ，子弹可能会偏（瞄准了却没打中、瞄歪却打中了）
> 
> 具体解释：
> - s'随机，r确定：坏枪+好子弹，枪不一定能准确瞄准靶子，但子弹一定能准确打到瞄准的位置
> - s'确定，r随机：好枪+坏子弹，枪一定能准确瞄准靶子，但子弹可能偏到其他位置
> - s'随机，r随机：坏枪+坏子弹，枪瞄的不准，子弹落点也不准
> - s'确定，r确定（确定性环境）：好枪+好子弹在

## 强化学习中的随机性：策略随机性 vs 环境随机性

强化学习中的随机性有两个来源：
- 策略随机性 $π(a|s)$ ：给定状态s，以根据某种概率分布随机生成动作a。（与之相对的是最优性策略：给定s，生成某个“最优”的a）
- 环境随机性 $P(s',r|s,a)$ ：给状态s和动作a，转移到的新状态s'或获得的奖励r有随机性（与之相对的是确定性环境，转移到的s'和r都确定）

标准的RL各种公式（例如贝尔曼方程）中，是按策略、环境都随机给出的。
但在LLM强化学习的语境下：
- s：prompt+已经生成的token；a：新token或新response序列
- 策略随机性：对于top-p解码，策略是随机的；对于贪婪解码，策略是确定的
- 环境随机性：s无随机性（给定s和a，s'=s拼接a）；r如果由规则判断则是确定的，如果是llm-as-judge可能有随机性（本文暂不考虑这种情况）

对于LLM的强化学习（策略随机、环境确定），比标准的强化学习（策略随机、环境随机）的设定更简单，因此各种公式会有一些简化。

## 马尔科夫性
todo

# 衍生概念：G、V、A、Q

## 轨迹trace（或称episode）τ
$τ=(s_1, a_1, r_1, s_2, a_2, r_2, s_3, ..., s_n, a_n, r_n, s_{n+1})$ 从起始到结束称为一个轨迹。
- 第t步的初始状态为 $s_t$ ， 执行动作 $a_t$ ， 此时立刻获得奖励 $r_t$ ，并转移到下个状态 $s_{t+1}$
- 一共执行t=1...n，共n次行动
- $τ \sim π(a|s)$ 代表从状态s开始，根据策略 π(a|s) 依次采样下一个动作，直到终局，所产生的轨迹，即“从概率π(a|s)采样的轨迹τ”
  - “从状态s开始”不代表s必须是第一个状态 $s_1$ ，它可以是任何一个中间状态 $s_t$ 。根据马尔科夫性， $s_t$ 之后的演进只和 $s_t$ 有关，与之前如何来到 $s_t$ 无关。

## 收益（gain）G 与折扣因子 $γ$
第t步的收益 $G_t = r_{t} + γ r_{t+1} + γ^2 r_{t+2} + ... = \sum_{i=0}^{\infty} γ^i r_{t+i}$
- 折扣因子 0≤γ≤1
- 含义：从第t步开始直到终局，累计reward之和（越未来的步骤通常越“不重要”，因此使用折扣γ衰减。γ=0 则只看当前动作的reward，γ=1 则每一步的reward都同等重要）
- G默认是到终局位置，但也可以计算k步收益： $G_t^{(k)} = r_{t} + γ r_{t+1} + ... + γ^{k-1} r_{t+k-1} = \sum_{i=0}^{k-1} γ^i r_{t+i}$ ，它在k步时序差分中会用到（本文不介绍）
- 收益（gain）有时也称作回报（return），两者在强化学习中含义相同，只是不同地方的术语有差异。这里称呼为gain，防止return和reward相似混淆

## 价值（value）函数 $V^π(s)$ ：
对于策略π，状态s的价值为：
```math
\begin{aligned}
V^π(s) 
& = \mathbb{E}_{τ \sim π(a|s)}[G_t | S_t=s]  \\
& = \mathbb{E}_{τ \sim π(a|s)}[r_{t} + γ r_{t+1} + γ^2 r_{t+2} + ... | S_t=s]
\end{aligned}
```
- 其中状态s执行动作a后，获得奖励r，状态变为s'
- **物理意义**：从状态s按策略π(a|s)采样直到终局，**预期**能累积到多少奖励。

几点注意事项：
- 这里大写的 $S_t$ 表示第t个状态的“占位符”，小写的 $s$ 代表状态的实际取值，可以类比“类vs实例”，正好也是类大写、实例小写
- $V^π(s)$ 和 s 处于序列的第几步无关，只和s与π有关
- 对于不同的策略π，同一个状态s的价值 $V^π(s)$ 可能不同
- $s_t$ 的价值是从 $s_t$ 的**后续**奖励 $r_t$ 开始算的，而不是从到达s时产生的奖励 $r_{t-1}$ 开始算的

### 辨析：奖励r、收益G、价值V
> - 奖励reward：第t步行动**实际产生**的**即时**“收获”，衡量**状态s+动作a**的好坏
> - 收益gain：从第t步到终局**实际产生**的**累计**“收获”，衡量**轨迹τ**的好坏
> - 价值V：即从s到终局时，**预期会产生**的**累计**“收获”，衡量**状态s**的好坏

### 为什么引入价值V（为什么只靠奖励r不够）？
- reward稀疏：有些场景，中间的过程没有奖励，只有终局时才能一次性结算奖励（LLM生成过程就很典型，生成完才能评估是否正确，单个token没有奖励）
- reward好坏 ≠ 动作/状态好坏：即刻的奖励高，可能后续的奖励少，不一定是个好动作/好状态。

因此reward是“局部的、即刻的”，价值V是“全局的、考虑后续的”，它能补齐reward的视角：
- reward稀疏：没有reward时，价值V依然能评估出状态的好坏，并且根据转移到下个状态的V评估动作好坏
- reward好坏 ≠ 动作/状态好坏：考虑对未来的预期，平衡即刻的reward数值

### 关于价值V本身的一些澄清：
**价值V是客观的 → 客观但不可直接获取 → 不可获取但可以估计 → 估计有误差 → 误差可以收敛**  
> 逐句解释：
> - 价值是客观的：给定策略π和状态s，价值 $V^π(s)$ 一定是个客观实际的值。（类比：概率分布的期望是个客观实际的值）
> - 客观但不可直接获取：reward和gain是可以直接获取的，但价值V是**期望**，它不能直接获取。（类比：样本的值能直接获取，但背后的期望不能直接获取）
> - 不可获取但可以估计：有大量真实数据时，可以构造统计量来近似估计价值 $V^π(s)$ 。（类比：真实数据的平均数是个人工构造的统计量，可以近似估计概率分布的期望）
>   - 动态规划法中可以为V随机赋值，并且迭代解不动点
>   - PPO中用神经网络计算V
>   - 蒙特卡洛中构造统计量估计V
> - 估计有误差：根据真实数据估计出的价值V，数值上和客观实际的V不一定恰好相等。（类比：数据的平均数≈概率的期望，但不一定恰好相等）
> - 误差可以收敛：数据足够多/迭代步数足够多，估计出的V和客观实际的V误差可忽略。（类比：数据的平均数≈概率的期望，在数据足够多时误差可忽略）

## 动作价值函数 $Q^π(s,a)$ 

对于策略π，在状态s下，动作a的价值为：
```math
\begin{aligned}
Q^π(s,a) 
& = \mathbb{E}_{τ \sim π(a|s)}[G_t | S_t=s, A_t=a]  \\
& = \mathbb{E}_{τ \sim π(a|s)}[r_{t} + γ r_{t+1} + γ^2 r_{t+2} + ... | S_t=s, A_t=a]
\end{aligned}
```
- 其中状态s执行动作a后，获得奖励r，状态变为s'
- **物理意义**：从状态s**明确执行动作a**后，未来再按策略π采样直到终局，**预期**能累积到多少奖励。
  - Q也要从s执行a之后产生的奖励开始计算，这点和V的计算一致

### Q和V的差异与联系：
- $Q^π(s,a)$ 衡量**状态s下动作a**的好坏，即状态s**明确采取动作a**后的预期收获（只是说在这一步采取动作a，后续动作还是按策略π采样）
- $V^π(s)$ 衡量**状态s**的好坏，即状态s按策略π随机动作，并且将这些**动作的效果累加**后的预期收获
  - 也就是**贝尔曼方程**： $V^π(s) = \sum_{a} π(a|s) Q^π(s, a)$
  - 贝尔曼方程反映了V和Q的关系
  - 贝尔曼方程还有很多其他形式，这里不展开
  - 这里介绍贝尔曼方程，只是为了解释V和Q的关系，GRPO不涉及贝尔曼方程


## 优势（advantage）函数 $A^π(s,a)$ 

在给定状态s下，动作a的优势为： $A^π(s,a) = Q^π(s,a) - V^π(s)$
- 借助贝尔曼方程辅助理解： $A^π(s,a) = Q^π(s,a) - \sum_{a} π(a|s) Q^π(s, a)$
  - 其中 $V^π(s) = \sum_{a} π(a|s) Q^π(s, a)$
- 从贝尔曼方程可以看出，优势 $A^π(s,a)$ 的**物理意义**：在状态s下，明确采取动作a，比按策略π随机采取一个动作，产生的**额外**的预期收益
  - $A^π(s,a)>0$ ：a是个好动作
  - $A^π(s,a)<0$ ：a是个坏动作
  - $A^π(s,a)=0$ ：a不好不坏
- 数学性质：给定状态s，所有动作a的收益 $A^π(s,a)$ 之和为0
  - 因为A衡量的是一个动作相比于所有动作的“相对值”，动作有好有坏，有的相对收益为正，就一定有的相对收益为负（相对收益、额外收益是一回事）
- 这个优势A和GPRO中的优势A是**同一个东西**，后面会推导。

## 拓展内容：贝尔曼方程（V←V、Q←Q、V←Q、Q←V 的互相递归计算）

推导蒙特卡洛、GRPO不需要贝尔曼方程，此处略。推导时序差分、PPO才需要。

<details>
<summary><b>点击展开贝尔曼方程的内容</b></summary>
  
**概率环境**：
```math
\begin{aligned}
V^π(s) & = \sum_{a} π(a|s) \sum_{s', r} P(s', r|s, a) [ r + γ V^π(s') ] \qquad \text{1. V计算V} \\
Q^π(s, a) & = \sum_{s', r} P(s', r|s, a) [ r + γ \sum_{a'} π(a'|s') Q^π(s', a') ] \qquad \text{2. Q计算Q} \\
V^π(s) & = \sum_{a} π(a|s) Q^π(s, a) \qquad \text{3. 用Q计算V} \\ 
Q^π(s, a) & = \sum_{s', r} P(s', r|s, a) [ r + γ V^π(s')] \qquad \text{4. 用V计算Q}
\end{aligned}
```

**确定性环境**：去掉所有 $P(s',r|s,a)$
```math
\begin{aligned}
V^π(s) & = \sum_{a} π(a|s) [ r + γ V^π(s') ] \qquad \text{1. V计算V} \\
Q^π(s, a) & = r + γ \sum_{a'} π(a'|s') Q^π(s', a') \qquad \text{2. Q计算Q} \\
V^π(s) & = \sum_{a} π(a|s) Q^π(s, a) \qquad \text{3. 用Q计算V} \\ 
Q^π(s, a) & = r + γ V^π(s') \qquad \text{4. 用V计算Q}
\end{aligned}
```

</details>  

# 蒙特卡洛：估计V/Q/A

前面介绍价值V时有解释：**价值V是客观的 → 客观但不可直接获取 → 不可获取但可以估计 → 估计有误差 → 误差可以收敛**，这对于Q和优势A也同样成立。

估计V、Q、A的方法很多，其中蒙特卡洛法MC是最直观的。蒙特卡洛的结果带入LLM的场景中做简化，可以直接推导出GRPO中优势A的公式（下一章介绍）

> 除了蒙特卡洛，估计V/Q/A还有其他方法。例如：
> - 动态规划：本质是为V/Q/A随机赋值后，迭代法求不动点。但只适合状态少、动作少的离散toy场景，此处不介绍
> - 时序差分TD：这个分支最终演化为PPO的优势估计（演化过程：时序差分 → 多步时序差分 → 广义优势估计GAE → PPO中的优势A），以后讲PPO时再介绍

蒙特卡洛的本质：生成一堆数据，拿到其中的reward，“直接计算” V/Q/A
  - 类比：采样一堆数据，用它们的均值“直接计算”分布的期望

## 回顾V/Q/A的定义：
```math
\begin{aligned}
G_t &= r_{t} + γ r_{t+1} + γ^2 r_{t+2} + γ^3 r_{t+3} + ...  \qquad \text{where  0≤γ≤1} \\
V^π(s) & = \mathbb{E}_{τ \sim π(a|s)}[G_t | S_t=s] \\
Q^π(s,a) & = \mathbb{E}_{τ \sim π(a|s)}[G_t | S_t=s, A_t=a] \\
A^π(s,a) &= Q^π(s,a) - V^π(s)
\end{aligned}
```

## 蒙特卡洛算法：
- 依据策略π，采样出大量 $τ=(s_1, a_1, r_1, s_2, a_2, r_2, s_3, ...)$ 轨迹，也就是rollout
- 对于某个状态 $s$ 和 动作 $a$ ：
  - 估计 $V^π(s)$ ：
    - 找到所有包含状态 $s$ 的轨迹，轨迹数量记为 $N(s)$
    - 对于每条这样的轨迹，截取从 $S_t = s$ 一直到结尾的轨迹片段，抽取其中的奖励 $r_{t}, r_{t+1}, ...$ ，计算该片段的gain： $G_t (S_t=s) = r_{t} + γ r_{t+1} + γ^2 r_{t+2} + ...$
    - 将所有这样轨迹的gain取平均： $\hat{V}^π(s)=\frac{\sum_{包含s的τ} G_t (S_t=s)}{N(s)}$
  - 估计 $Q^π(s,a)$ ：
    - 找到所有包含状态-动作对 $(s,a)$ 的轨迹，轨迹数量记为 $N(s, a)$
    - 对于每条这样的轨迹，截取从 $(S_t, A_{t}) = (s,a)$ 一直到结尾的轨迹片段，抽取其中的奖励 $r_{t}, r_{t+1}, ...$ ，计算该片段的gain： $G_t(S_t=s, A_t=a) = r_{t} + γ r_{t+1} + γ^2 r_{t+2} + ...$
    - 将所有这样轨迹的gain取平均： $\hat{Q}^π(s,a)=\frac{\sum_{包含(s,a)的τ} G_t (S_t=s, A_t=a)}{N(s, a)}$
  - 估计 $A^π(s,a)$ ：
    - $\hat{A}^π(s,a) = \hat{Q}^π(s,a) - \hat{A}^π(s)$

上述三个统计量 $\hat{V}^π, \hat{Q}^π, \hat{A}^π$ ，就是对 $V^π, Q^π, A^π$ 的估计。这个估计高方差、无偏差（与此相对，时序差分对 $V^π, Q^π, A^π$ 的估计是低方差、有偏差的。以后PPO中介绍，此处略）


# GRPO中的优势A：简化环境下的蒙特卡洛

回到强化学习训练LLM的场景。我们按如下方法定义强化学习问题：

## 建模：sequence级建模（只有一问一答）
- 只有一个初始状态、一个结束状态、一个动作
- 完整轨迹：只有 $τ=(s, a, r, s')$
  - 初始状态 $s$ ：prompt $x$
  - 动作 $a$ ：LLM生成的完整response $y$ ，（动作空间是无限的）
  - 奖励 $r$ ：针对完整response $y$ 的奖励（例如答案是否正确）
  - 结束状态 $s'$ ：prompt $x$ + response $y$
  - 策略 $π(a|s)$ ： LLM生成response y的概率 $P(y | x)$
  - 环境的随机性 $P(s',a'|s,a)$ ：环境是确定的，没有随机性

## 蒙特卡洛估计
- rollout：取一个prompt $x$ ，生成N个不同的response $\{y_1, ..., y_N\}$ ，对应N种不同的动作
  - 记 $y_i$ 为第i个rollout， $r_i$ 为对应的奖励
- 各种量的对应：
  - 第i个轨迹 $τ_i = (s,a,r,s') = (x, y_i, r_i, x+y_i)$
    - 初始状态 $s$ ：prompt $x$
    - 动作 $a$ ：第i个完整response $y_i$
    - 奖励 $r$ ： $y_i$ 的奖励 $r_i$
    - 结束状态 $s'$ ：prompt $x$ + response $y_i$
  - 包含 $s=x$ 的轨迹数量 $N(x)$ ：
    - 一共生成N个轨迹，每个轨迹的s都是prompt $x$
    - $N(s)=N$
  - 包含 $(s,a)=(x,y_i)$ 的轨迹数量 $N(s, a)$ ：
    - 每个 (prompt, response) 对只出现一次
    - $N(s, a)=1$
  - 轨迹片段的收益 $G_t(τ_i)$ ：
    - 在每一个轨迹 $τ_i$ 上，从第t步直到结束的收益 $G_t(τ_i) =r_{i,t} + γ r_{i,t+1} + γ^2 r_{i,t+2} + ...$
    - 因为只有一步动作、一个最终奖励，因此只有 $r_{i,1}=r_{i}$ ，不存在 $r_{i,2}, r_{i,3}，...$
    - 完整轨迹 $G_t(τ_i)=r_{i,1}=r_i$ ，不需要考虑折扣因子 γ
    - 轨迹片段 $G_t(s=x)=G_t(s=x, a=y_i)=G_t(τ_i)=r_i$
- 带入，估计V、Q、A：
  - $\hat{V}^π(x) = \hat{V}^π(s=x) = \frac{\sum_{包含x的τ_i} G_t (s=x)}{N(s=x)} = \frac{\sum_{i=1}^{N} r_i}{N} = μ$ ，即奖励 $r_1, ..., r_N$ 的平均值 $μ$
  - $\hat{Q}^π(x, y_i) = \hat{Q}^π(s=x, a=y_i)=\frac{\sum_{包含(x,y_i)的τ_i} G_t (s=x, a=y_i)}{N(s=x,a=y_i)} = \frac{r_i}{1}=r_i$ ，即奖励 $r_i$ 本身
  - $\hat{A}^π(x, y_i) = \hat{Q}^π(x, y_i) - \hat{V}^π(x) = r_i - μ$

观察GRPO公式中的优势： $A(x, y_i) = \frac{ r_i - μ}{σ}$  ，可以发现就是蒙特卡洛的 $\hat{A}^π(x, y_i)= r_i - μ$ 
  - 唯一的差别是分母 σ，这是为了控制方差范围做的scale（蒙特卡洛本来就有方差大的缺点），属于trick，不影响本质
  - 这里也能看出为什么计算 $A(x, y_i)$ 必须基于相同的prompt：因为 $V^π(s)$ 必须基于同一个起始状态s才有意义（这里是prompt x）

## token级建模：单个token的优势A

以上介绍的是将完整response视为一个动作，计算出的优势A。但在GRPO中，优势A是针对每个token的，以下将推导单个token的优势。

### 形式化
- 问题1：如果将一系列动作合并为一个“大动作”，则大动作的优势是多少？  
  - 结论：记单个动作的优势是 $A_i$ , “大动作”的总优势是 $A_{\text{total}}$ , 则 $A_{\text{total}} = A_1 + γ A_2 + γ^2 A_3 + ...$ 即折扣衰减后的优势之和

- 问题2：反过来，已知 “大动作”的总优势是 $A_{\text{total}}$ ，求每个动作的优势 $A_i$
  - 引入假设：**每一步动作同等重要** （即一个轨迹上所有动作的优势 $A_i$ 相等）
  - 结论：简单的等比数列求和，其中 $n$ 为轨迹τ上的动作数量

```math
A_i = 
\begin{cases}
\frac{1}{n} A_{\text{total}} & \text{if } γ = 1, \\[1em]
\frac{(1-γ)}{1-γ^{n}} A_{\text{total}} & \text{if } γ \neq 1.
\end{cases}
```

### 对应到LLM
- 把每个token视为一个动作，则整个response就是一个大动作
- 已知整个response的总优势 $A_{\text{total}} = \frac{r-μ}{σ}$ ，倒推每个token的优势 $A_i$
- 引入假设：response $y$ 中的每个token同等重要
- 结果：
  - γ=1，即无衰减：  $A_i = \frac{1}{|y|} A_{\text{total}} = \frac{1}{|y|} \cdot \frac{r-μ}{σ}$ ，其中 $|y|$ 是y的长度
  - γ=0，即每个token只关心自己：  $A_i = A_{\text{total}} = \frac{r-μ}{σ}$ 
- GRPO通常采用γ=1的设置


## 总结：GRPO的优势A的本质

**sequence级建模**：
- 把prompt当成起始状态s
- 把整个response当成一个动作a
- 以同一个prompt的多个rollout作为样本池
- 用蒙特卡洛法估计出的优势A
- 对应GSPO中整个response的优势A

**token级建模**：
- 把response中的每个token视为一个动作a
- 引入两个假设：
  - 折扣因子γ=0（即每个token只关心自己）
  - 每个token同等重要，即它们的优势相同
- 则每个token的优势A，就是整个response的优势A
- 对应GRPO中每个token的优势A

> 这也是为什么GSPO比GRPO更“自然”：GRPO额外引入了两个假设

以上介绍了GRPO公式中优势 $A$ 的来源，接下来将介绍RL的目标函数，如何对目标函数求导，以及将RL目标函数和优势A组合，得到GRPO的公式。

# RL的目标函数（原始形式）

RL的目标函数为： $J(θ) = \mathbb{E}_{s_1 \sim D(s_1), τ \sim π_θ(τ|s_1)}[G(τ)]$  

> 该结果对任何强化学习都成立，无论一步决策还是多步决策、把token视为一个动作还是把完整response视为一个动作  
> 以下默认将每个token视为一个动作，对应GRPO的结果

## 物理含义：
- 极大化轨迹的期望gain $G(τ)$ 。（注：如果是梯度下降，则目标函数取反变成 $-J(θ)$ 即为损失函数）

## 公式拆解：
- $s_1 \sim D(s_1)$: 基于初始状态 $s_1$ 的分布 $D(s_1)$ 采样 $s_1$
  > 对应到LLM：在数据集 D 中采样 prompt $x$
- $τ \sim π_θ(s_1)$ ：给定初始状态 $s_1$ ，通过策略 $π_θ(τ|s_1)$ 采样出完整轨迹 $τ$
  > 对应到LLM：给定 prompt $x$ ，用LLM 生成response $y$  
  > 这里将每个token视为一个动作  
  > $π_θ(τ|s_1)=π_θ(y|x)=π_θ(y_1|x)π_θ(y_2|x,y_1)π_θ(y_3|x,y_{1:2})...$  
  > 其中 $y_i$ 是 $y$ 的第 i 个token, $π_θ$ 是LLM，θ是模型权重
- $G(τ)$ ：轨迹 $τ$ 的收益（gain），$G(τ)=r_1 + γ r_2 + γ^2 r_3 + ...$ ，$r_t$ 为第t步动作的reward
  > 对应到LLM：
  > 令折扣因此 γ=1，即不进行折扣，则 $G(τ)=r_1 + r_2 + r_3 + ...$  
  > 每个token没有中间reward，只有最终结果reward，则 $G(τ)=0 + 0 + ... + r_{\text{final}=}r_{\text{final}}$
- $\mathbb{E}_{s_1 \sim D(s_1), τ \sim π_θ(τ|s_1)}[G(τ)]$ ：对采样出的所有 $s_1$ 和 $τ$ 取平均
  > 对应到LLM：  
  > $\mathbb{E}_{s_1 \sim D(s_1), τ \sim π_θ(τ|s_1)}[G(τ)]=E_{x \sim D(x), y \sim π_θ(y|x)}[r_{\text{final}}]$  
  > - 采样出 M 个prompt $x^1,...,x^{M}$  
  > - 每个prompt 采样 N 个 response: $y^{1,1} ..., y^{1,N}, ..., y^{M,N}$  
  > - 一共M\*N条轨迹τ：$(x^1, y^{1,1}), ..., (x^1, y^{1,N}), ..., (x^M, y^{M,N})$  
  > - 对M\*N个样本的 $G(τ)=r_{\text{final}}$ 取平均

  

# 对目标函数求导 & 目标函数的变形

此处先给结论，之后再做推导。该结论实际是【**策略梯度定理**】

## 变形后“实际使用”的目标函数： 
```math
\begin{aligned}
J(θ) 
&= \mathbb{E}_{s_1 \sim D(s_1), τ \sim π_θ(τ|s_1)_{.detach}}\left[G(τ) \frac{π_θ(τ|s_1)}{π_θ(τ|s_1)_{.detach}} \right] \qquad 理论结果 \\
&≈ \frac{1}{M} \frac{1}{N} \sum_{采样M个初始状态s_1} \sum_{每个s_1采样N个轨迹τ} \left[G(τ) \frac{π_θ(τ|s_1)}{π_θ(τ|s_1)_{.detach}}\right] \qquad 实际计算图 \\
&≈ \frac{1}{M} \frac{1}{N} \sum_{采样M个初始状态s_1} \sum_{每个s_1采样N个轨迹τ} [G(τ)]  \qquad 数值结果
\end{aligned}
```

## 对目标函数求导：
```math
\begin{aligned}
\nabla_θ J(θ) 
&= \mathbb{E}_{s_1 \sim D(s_1), τ \sim π_θ(τ|s_1)_{.detach}}\left[G(τ) \frac{π_θ(τ|s_1)}{π_θ(τ|s_1)_{.detach}} \nabla_θ \log π_θ(τ|s_1) \right] \qquad 理论结果 \\
&≈ \frac{1}{M} \frac{1}{N} \sum_{采样M个初始状态s_1} \sum_{每个s_1采样N个轨迹τ} \left[G(τ) \frac{π_θ(τ|s_1)}{π_θ(τ|s_1)_{.detach}} \nabla_θ \log π_θ(τ|s_1)\right] \qquad 实际计算图 \\
&≈ \frac{1}{M} \frac{1}{N} \sum_{采样M个初始状态s_1} \sum_{每个s_1采样N个轨迹τ} [G(τ) \cdot \nabla_θ \log π_θ(τ|s_1)]  \qquad 数值结果
\end{aligned}
```

> ### 如何理解 $π_θ(τ|s_1)_{.detach}$ ？
> - detach指的就是pytorch中的detach操作，即把某个网络分离出来不计算梯度  
> - 引入detach，是为了强行让 $π_θ(τ|s_1)$ 能计算梯度，后面会介绍。 $\frac{π_θ(τ|s_1)}{π_θ(τ|s_1)_{.detach}}$ 类似于gumbel-softmax中 $\text{OneHot}(s) - s_{.detach} + s$ ，在计算图中不能省略
> - $\frac{π_θ(τ|s_1)}{π_θ(τ|s_1)_{.detach}}$ 本质上是一种**重要性采样**，下一章介绍。
> - 在LLM强化学习的语境下， $π_θ(τ|s_1)$ 和 $π_θ(τ|s_1)_{.detach}$ 有具体的对应物： 

|             | $π_θ(τ\|s_1)$            | $π_θ(τ\|s_1)_{.detach}$                        |
|-------------|--------------------------|------------------------------------------------|
| 计算框架    | fsdp 等训练框架部署的 LLM  | vllm 等推理框架部署的 LLM（有针对性加速手段）     |
| 用处        | 计算梯度                  | rollout 和 evaluate                            |
| 参数更新方式 | 梯度下降直接更新          | 将更新后的 $π_θ(τ\|s_1)$ 参数复制过来（即 detach）|

> ### 如何理解两个“约等于≈”？
> - 理论结果→实际计算图：把期望 $\mathbb{E}$ 变成采样求平均 $\frac{1}{M} \frac{1}{N} \sum_{M} \sum_{N}$ ，本身就有误差
> - 实际计算图→数值结果：因为训练、推理框架的差异，$\frac{π_θ(τ|s_1)}{π_θ(τ|s_1)_{.detach}}$ 可能不等于1

> ### 策略梯度定理的两种写法
> 很多教程中，策略梯度定理的写法更加简洁：   
> $\nabla_θ J(θ) = \mathbb{E}_{s_1 \sim D(s_1), τ \sim π_θ(τ|s_1)} \left[G(τ) \nabla_θ \log π_θ(τ|s_1) \right]$  
>   
> 但本文的写法更贴近本质，且和现实RL框架的代码吻合：  
> $\nabla_θ J(θ) = \mathbb{E}_{s_1 \sim D(s_1), τ \sim π_θ(τ|s_1)_{.detach}}\left[G(τ) \frac{π_θ(τ|s_1)}{π_θ(τ|s_1)_{.detach}} \nabla_θ \log π_θ(τ|s_1) \right]$


# 公式推导

## 直接求导的困境

观察RL的目标函数 $J(θ) = \mathbb{E}_{s_1 \sim D(s_1), τ \sim π_θ(τ|s_1)}[G(τ)]$ 
- 该函数实际没办法“直接求导”
- 把期望展开成采样取平均：
```math
\begin{aligned}
J(θ) 
& = \mathbb{E}_{s_1 \sim D(s_1), τ \sim π_θ(τ|s_1)}[G(τ)] \\
& ≈ \frac{1}{M} \frac{1}{N} \sum_{采样M个初始状态s_1} \sum_{每个s_1采样N个轨迹τ} [G(τ)]
\end{aligned}
```
- 发现：展开后的表达式完全不含参数θ，因此无法对θ求导
> 其中 $G(τ)$ 用规则计算轨迹的gain，因此不含参数θ
- 问题的本质：参数θ在“分布” $π_θ(τ|s_1)$ 上，而不在“打分函数” $G(τ)$ 上 

以下先岔开一支，介绍机器学习中的两种目标函数，从而更好地理解这个目标函数为什么无法直接求导。

## 两种机器学习目标函数

机器学习中有两种目标函数：
- 参数θ在打分函数上： $J_θ = \mathbb{E}_{x \sim p(x)} [f_θ(x)]$
- 参数θ在分布上： $\qquad J_θ = \mathbb{E}_{x \sim p_θ(x)} [f(x)]$

目标：极大化/极小化采样出的样本的打分期望
- $x$ ：采样的样本
- $p(x)$ 、$p_θ(x)$ ：样本的概率分布
- $f(x)$ 、$f_θ(x)$ ：对样本的打分函数

绝大多数机器学习问题，都是参数θ在打分函数 $f(x)$ 上，分布 $p(x)$ 不含参数  
- 这种目标函数可以直接求导：  
- $\nabla_θ J_θ = \nabla_θ \mathbb{E}_{x \sim p(x)} [f_θ(x)] = \mathbb{E}_{x \sim p(x)} [ \nabla_θ f_θ(x)]$
> 例：LLM的SFT（极大化gold response的概率）  
> $J_θ = \mathbb{E}_{x \sim p(x)} [f_θ(x)] = \mathbb{E}_{(x,y) \sim D(x,y)} [π_θ(y|x)]$  
> $\nabla_θ J_θ = \mathbb{E}_{(x,y) \sim D(x,y)} [ \nabla_θ π_θ(y|x)]$
> - 样本 $x → (x,y)$ ： 数据集中提供的prompt $x$ 和response $y$
> - 分布 $p(x) → D(x,y)$ ：数据分布，相当于包含配对 prompt-response 的完整数据集
> - 打分函数 $f_θ(x) → π_θ(y|x)$ ：给定prompt x，生成指定response y 的概率

然而有少数机器学习问题，参数θ在分布 $p(x)$ 上，打分函数 $f(x)$ 反而不含参数。 

> 例：LLM的GRPO（极大化rollout response的收益）  
> $J(θ) =  \mathbb{E}_{x \sim p_θ(x)} [f(x)] = \mathbb{E}_{x \sim D(x), y \sim π_θ(y|x)}[G(x,y)]$  
> - 样本 $x → (x,y)$ ： prompt $x$ + response $y$
> - 分布 $p_θ(x) → p_θ(x,y)=D(x) \cdot π_θ(y|x)$ ：从数据集采样 prompt $x$ ，从LLM采样 response $y$
> - 打分函数 $f(x) → G(x,y)$ ：利用规则计算reward（以及相应的gain），不含参数
> - 和SFT的区别：
>   - SFT提供了gold response，且LLM只生成gold response，不生成负例
>   - GRPO不提供gold response（只提供了一个参考标准），LLM随机rollout各种response，其中有正例有负例
 

另一个典型的对分布 $p_θ(x)$ 求导的问题：VAE的解码器
> 给定一个噪声 $z$ ，把它随机映射成图像 $x'$ ，极小化它和原始图像 $x$ 的误差
> - 带参数的分布 $p_θ(x) → p_θ(x'|z)$ ，噪声 $z$ 映射出的图像 $x'$ 不是唯一的，而是一个图像分布
> - 样本 $x → x'$ ，从映射出的分布中随机采样一样图像 $x'$
> - 不带参数的打分函数 $f(x) → f(x')$ ： 例如MSE，直接衡量 $x$ 和 $x'$ 的差距
> - 以上只是个不严格的介绍，便于理解。VAE不是本文重点

后一类参数在 $p_θ(x)$ 上的机器学习任务，本质上是要优化一个带参数的采样。
- 传统机器学习问题：给定输入x，输出y是固定的（例如分类器）
- 带参数采样问题：给定输入x，输出y是随机的（例如RL中给定初始状态，随机采样一个动作）

带参数采样的目标函数不能直接求导：
- $\nabla_θ J_θ = \nabla_θ \mathbb{E}_{x \sim p_θ(x)} [f(x)] ≠ \mathbb{E}_{x \sim p_θ(x)} [ \nabla_θ f(x)]$
- 因为 $f(x)$ 不含 θ， $\nabla_θ f(x) = 0$

正确的求导结果：
```math
\begin{aligned}
\nabla_θ J_θ 
& = \nabla_θ \mathbb{E}_{x \sim p_θ(x)} [f(x)] \\
& = \mathbb{E}_{x \sim p_θ(x)} [f(x) \nabla_θ \log p_θ(x)] \qquad 正确结果（简写） \\
& = \mathbb{E}_{x \sim p_θ(x)_{.detach}}\left[f(x) \frac{p_θ(x)}{p_θ(x)_{.detach}} \nabla_θ \log p_θ(x) \right]  \qquad 正确结果（本质）
\end{aligned}
```

> 下面将推导这个正确结果

## 推导：对带参数的采样 $p_θ(x)$ 求导

- 问题：分布 $p_θ(x)$ 含参数，打分函数 $f(x)$ 不含参数，直接求导 $\nabla_θ f(x)=0$
- 解决思路：把参数θ从分布挪到打分函数上
- 常用做法：重参数化（例：VAE对高斯分布重参数化，gumbel-softmax对离散采样重参数化）
- 此处借助重参数化+重要性采样+对数导数技巧，推导 $\nabla_θ J_θ$ 的公式 

几个数学技巧（这里直接当引理给出）
### 技巧1：“广义的”重参数化方法
> 用途：把参数θ从分布挪到打分函数上  

对于目标函数 $J_θ = \mathbb{E}_{x \sim p_θ(x)} [f(x)]$  
如果我们能找到另一组无参数分布 $q(x)$ 和含参数的打分函数 $g_θ(x)$ ，使得：  
- $J_θ = \mathbb{E}_{x \sim p_θ(x)} [f(x)] = \mathbb{E}_{x \sim q(x)} [g_θ(x)]$
- 分布 $p_θ(x)$ 和 打分函数 $g_θ(x)$ 使用同一组参数θ

这样就把参数θ从分布转移到打分函数上了，从而能对 $J_θ$ 直接求导：  
$$
\nabla_θ J_θ = \mathbb{E}_{x \sim q(x)} [ \nabla_θ g_θ(x)]
$$

> **“广义的”重参数化**这个名字是我编的，没见过类似写法，但直觉上是对的  
> 标准的重参数化方法不长这样，但和本文推导无关，这里就不介绍了  

### 技巧2：重要性采样（下一章会更详细探讨）
> 用途：构造上面的 $q(x)$ 和 $g_θ(x)$

- 需求：在用采样求期望 $\mathbb{E}_{x \sim p(x)} [f(x)]$ 时，如果从原始分布 $p(x)$ 采样 $x$ 较为困难，但从另一个分布 $q(x)$ 采样 $x$ 较为容易，如何用从 $q(x)$ 采样的 $x$ 估计在 $p(x)$ 上的期望？ 
- 公式： $\mathbb{E}_{x \sim p(x)} [f(x)] = \mathbb{E}_{x \sim q(x)} \left[ \frac{p(x)}{q(x)} f(x) \right]$
> - 证明： $\mathbb{E}_{x \sim p(x)} [f(x)] = ∫ p(x)f(x)dx = ∫ q(x) \left[\frac{p(x)}{q(x)} f(x)\right]dx = \mathbb{E}_{x \sim q(x)} \left[ \frac{p(x)}{q(x)} f(x) \right]$
> - LLM中的实际场景：
>   - $x$ ：采样出的response
>   - $p(x)$ ：用训练框架fsdp部署的LLM，用于计算梯度，效率低
>   - $q(x)$ ：用推理框架vllm部署的LLM，用于rollout，效率高
>   - 成本考量：从 $q(x)$ 中采样，计算概率 $q(x)$ 和 $p(x)$ ，加在一起的成本比直接从 $p(x)$ 中采样还低
>   - 其中从 $p(x)$ 中采样 $x$ 成本很高，但给定采样好的 $x$ 计算它的概率 $p(x)$ 成本相对不高
>   - 这也是投机解码的原理：用低成本LLM采样response，在高成本LLM上验证response的概率

### 技巧3：对数导数技巧
> 用途：有利于实际计算时的数值稳定性

- 公式： $\nabla f(x) = f(x) \nabla \log f(x)$

> - 证明： $\nabla \log f(x) = \frac{\nabla f(x)}{f(x)} \Rightarrow f(x) = f(x) \nabla \log f(x)$


### 整合 & 公式推导

首先依据重参数化，我们要找到另一组无参数分布 $q(x)$ 和含参数的打分函数 $g_θ(x)$ ，使得：  
- $J_θ = \mathbb{E}_{x \sim p_θ(x)} [f(x)] = \mathbb{E}_{x \sim q(x)} [g_θ(x)]$
- 分布 $p_θ(x)$ 和 打分函数 $g_θ(x)$ 使用同一组参数θ

这样的 $q(x)$ 和 $g_θ(x)$ 可以借助重要性采样来构造：
- $q(x) = p_θ(x)_{.detach}$ ，其中detach后就相当于不含参数了
- $g_θ(x) = \frac{p_θ(x)}{p_θ(x)_{.detach}} f(x)$

$q(x)$ 和 $g_θ(x)$ 显然满足条件：  

$$
J_θ = \mathbb{E}_{x \sim p_θ(x)} [f(x)] = \mathbb{E}_{x \sim p_θ(x)_{.detach}} \left[ \frac{p_θ(x)}{p_θ(x)_{.detach}} f(x) \right] = \mathbb{E}_{x \sim q(x)} [g_θ(x)]
$$

这样就成功把参数θ从分布转移到了打分函数上，可以正常求导了。

```math
\begin{aligned}
\nabla_θ J_θ 
&= \mathbb{E}_{x \sim q(x)} [ \nabla_θ g_θ(x)] \\
&= \mathbb{E}_{x \sim p_θ(x)_{.detach}} \left[ \frac{ \nabla_θ p_θ(x)}{p_θ(x)_{.detach}} f(x) \right] \\
&= \mathbb{E}_{x \sim p_θ(x)_{.detach}} \left[ \frac{ p_θ(x)}{p_θ(x)_{.detach}} f(x) \nabla_θ \log p_θ(x) \right]  \qquad 对数导数技巧 \\
\end{aligned}
```

对应到RL的场景：
- 样本 $x$ ： 初始状态 $s_1$, 轨迹 $τ$
- 有参数概率采样 $x \sim p_θ(x)$ ：从数据集采样 $s_1 \sim D(s_1)$ （这部分没有参数）, 从策略采样 $τ \sim π_θ(τ|s_1)$
- 无参数打分函数 $f(x)$ ： 轨迹收益 $G(τ)$

带入：
```math
\begin{aligned}
J(θ) 
&= \mathbb{E}_{s_1 \sim D(s_1), τ \sim π_θ(τ|s_1)_{.detach}}\left[G(τ) \frac{π_θ(τ|s_1)}{π_θ(τ|s_1)_{.detach}} \right] \qquad 理论结果 \\
&≈ \frac{1}{M} \frac{1}{N} \sum_{采样M个初始状态s_1} \sum_{每个s_1采样N个轨迹τ} \left[G(τ) \frac{π_θ(τ|s_1)}{π_θ(τ|s_1)_{.detach}}\right] \qquad 实际计算图 \\
&≈ \frac{1}{M} \frac{1}{N} \sum_{采样M个初始状态s_1} \sum_{每个s_1采样N个轨迹τ} [G(τ)]  \qquad 数值结果
\end{aligned}
```

```math
\begin{aligned}
\nabla_θ J(θ) 
&= \mathbb{E}_{s_1 \sim D(s_1), τ \sim π_θ(τ|s_1)_{.detach}}\left[G(τ) \frac{π_θ(τ|s_1)}{π_θ(τ|s_1)_{.detach}} \nabla_θ \log π_θ(τ|s_1) \right] \qquad 理论结果 \\
&≈ \frac{1}{M} \frac{1}{N} \sum_{采样M个初始状态s_1} \sum_{每个s_1采样N个轨迹τ} \left[G(τ) \frac{π_θ(τ|s_1)}{π_θ(τ|s_1)_{.detach}} \nabla_θ \log π_θ(τ|s_1)\right] \qquad 实际计算图 \\
&≈ \frac{1}{M} \frac{1}{N} \sum_{采样M个初始状态s_1} \sum_{每个s_1采样N个轨迹τ} [G(τ) \cdot \nabla_θ \log π_θ(τ|s_1)]  \qquad 数值结果
\end{aligned}
```

这就是**策略梯度定理**的详细推导版本

# 实际常用形式

## 把轨迹展开为动作

前面的策略梯度定理中使用的是轨迹τ的概率 $π_θ(τ|s_1)$ ，实际应用中通常会进一步展开为每个动作的概率 $π_θ(a_t|s_t)$ 。

在确定性环境下：
```math
\begin{aligned}
\nabla_θ \log π_θ(τ|s_1)
&= \nabla_θ \log \prod_{t=1}^{|τ|} π_θ(a_t|s_t)  \qquad |τ|代表轨迹的步骤数 \\
&= \sum_{i=t}^{|τ|}  \nabla_θ \log π_θ(a_t|s_t)
\end{aligned}
```
> 在非确定环境下也成立，但和LLM无关，这里不介绍了

带入：
```math
\begin{aligned}
\nabla_θ J(θ) 
&≈ \frac{1}{M} \frac{1}{N} \sum_{采样M个初始状态s_1} \sum_{每个s_1采样N个轨迹τ} [G(τ) \cdot \nabla_θ \log π_θ(τ|s_1)]  \\
&= \frac{1}{M} \frac{1}{N} \sum_{采样M个初始状态s_1} \sum_{每个s_1采样N个轨迹τ} \left[ \sum_{t=1}^{|τ|} G(τ) \cdot \nabla_θ \log π_θ(a_t|s_t) \right] \\
&≈ \mathbb{E}_{s_0 \sim D(s_0), τ \sim π_θ(τ|s_0)} \left[ \sum_{t=1}^{|τ|} G(τ) \cdot \nabla_θ \log π_θ(a_t|s_t) \right]
\end{aligned}
```

> 这里只在求导版本 $\nabla J$ 下展开，目标函数 $J$ 没有log不能这么展开  
> GRPO的目标函数实际是用 $\nabla J$ 倒推回去的一个“代理目标函数” $\hat{J}$  
> $\hat{J}$ 和 $J$ 本质上不是同一个函数，但恰好 $\nabla J = \nabla \hat{J}$ ，且 $J$ 和 $\hat{J}$ 的极值点相同
> 最后一步的约等于：如果忽略 $π_θ(τ|s_0)$ 和 $π_θ(τ|s_0)_{.detach}$ 的数值差异，可以做这种“形式简化”

## 导数的变体：替换 $G(τ)$

实际应用中，一般会把轨迹收益 $G(τ)$ 替换成别的东西，公式还是成立的。
> 这部分的推理很复杂，就不管了。但公式看起来很符合直觉


```math
\begin{aligned}
\nabla_θ J(θ)
&≈ \frac{1}{M} \frac{1}{N} \sum_{采样M个初始状态s_1} \sum_{每个s_1采样N个轨迹τ} \left[ \sum_{t=1}^{|τ|} G(τ) \cdot \nabla_θ \log π_θ(a_t|s_t) \right]  \\
&= \frac{1}{M} \frac{1}{N} \sum_{采样M个初始状态s_1} \sum_{每个s_1采样N个轨迹τ} \left[ \sum_{t=1}^{|τ|} G_t(τ) \cdot \nabla_θ \log π_θ(a_t|s_t) \right]  \\
&= \frac{1}{M} \frac{1}{N} \sum_{采样M个初始状态s_1} \sum_{每个s_1采样N个轨迹τ} \left[ \sum_{t=1}^{|τ|} Q(s_t, a_t) \cdot \nabla_θ \log π_θ(a_t|s_t) \right]  \\
&= \frac{1}{M} \frac{1}{N} \sum_{采样M个初始状态s_1} \sum_{每个s_1采样N个轨迹τ} \left[ \sum_{t=1}^{|τ|} A(s_t, a_t) \cdot \nabla_θ \log π_θ(a_t|s_t) \right]
\end{aligned}
```

其中：
- 替换成从第t步开始的轨迹片段收益 $G_t(τ)$ ： REINFORCE算法，$G_t(τ)$ 从实际数据中收集
- 替换成Q函数 $Q(s_t, a_t)$ ：Actor-critic算法，Q函数用一个critic网络来计算
- 替换成优势 $A(s_t, a_t)$ ：PPO/GRPO等现代方法。PPO用一个critic网络来计算，GRPO直接对数据归一化计算

其中替换成优势A训练最稳定，因此现代方法一般使用A。
公式理解：用优势A控制梯度的方向和大小
- A>0：好动作，沿梯度方向走
- A<0：坏动作，逆着梯度方向走
- A>0且数值很大：非常好的动作，沿梯度方向走一大步

> 这里也能看出为什么引入 $\nabla_θ \log π_θ(τ|s_1)$ 来代替 $\nabla_θ π_θ(τ|s_1)$ :
> - 轨迹拆分成动作：通过log把动作概率相乘变成相加
> - logits： $\log π_θ(a_t|s_t)$ 实际是每个动作的logits，而神经网络给每个token的打分在softmax之前天然就是logits，省略一步softmax有利于数值稳定
> - 约分： $\frac{\nabla_θ π_θ(τ|s_1)}{π_θ(τ|s_1)_{.detach}} =\frac{π_θ(τ|s_1)}{π_θ(τ|s_1)_{.detach}} \nabla_θ \log π_θ(τ|s_1) ≈ \nabla_θ \log π_θ(τ|s_1)$ ，在不考虑训练-推理的数值差异时，可以约分

## 目标函数的变体：动作优势之和

如果把RL的目标函数从轨迹τ的收益 $G(τ)$ 改成动作优势之和:

$$
J(θ) = \mathbb{E}_{s_1 \sim D(s_1), τ \sim π_θ(τ|s_1)} \left[ \sum_{t=1}^{|τ|} γ^{t-1} A(s_t, a_t) \right]
$$

当折扣因子γ=1，即无衰减时：

$$
J(θ) = \mathbb{E}_{s_1 \sim D(s_1), τ \sim π_θ(τ|s_1)} \left[ \sum_{t=1}^{|τ|}A(s_t, a_t) \right]
$$


其导数恰好和对 $\mathbb{E}_{τ} [G(τ)]$ 求导的结果相同：

$$
\nabla_θ J(θ) = \mathbb{E}_{s_1 \sim D(s_1), τ \sim π_θ(τ|s_1)} \left[ \sum_{t=1}^{|τ|} A(s_t, a_t) \cdot \nabla_θ \log π_θ(a_t|s_t) \right]
$$

> 以上是简化形式，完整形式是：
> $J(θ) = \mathbb{E}_{s_1 \sim D(s_1), τ \sim π_θ(τ|s_1)_{.detach}} \left[ \sum_{t=1}^{|τ|} γ^{t-1} A(s_t, a_t) \frac{π_θ(a_t|s_t)}{π_θ(a_t|s_t)_{.detach}} \right]$
> $\nabla_θ J(θ) = \mathbb{E}_{s_1 \sim D(s_1), τ \sim π_θ(τ|s_1)_{.detach}} \left[ \sum_{t=1}^{|τ|} A(s_t, a_t) \frac{π_θ(a_t|s_t)}{π_θ(a_t|s_t)_{.detach}} \nabla_θ \log π_θ(a_t|s_t) \right]$
> 如果忽略训练、推理的差异，即认为在数值上 $π_θ(a_t|s_t)=π_θ(a_t|s_t)_{.detach}$ ，得到的就是简化形式

**这实际就是最简单版本的GRPO公式。**

# 最简单版本的GRPO公式

## 前提设置
- token级建模，每个token是一个动作
- 衰减因子γ=1
- 每个token同等重要
- 目标函数：极大化response中token优势之和的期望（此设置下等价于极大化轨迹收益）
- 优势估计方式：用同一个prompt rollout出一批数据后，做蒙特卡洛估计
- 不考虑KL散度
- on-policy，即rollout出一批数据只用来更新一次梯度，用后即丢弃（因此没有旧策略 $π_{old}$ ）
- 训练、推断的LLM用两个不同框架部署

## 符号约定：
  - 从数据集 $D(x)$ 采样 N 个prompt $x$  （为了简化公式，不引入角标）
  - 为每个 $x$ rollout G 个response $\{y_i\}_{i=1}^G$
  - $\{y_i\}_{i=1}^G$ 对应的reward为 $\{r_i\}_{i=1}^G$
  - $y_i$ 长度为 $|y_i|$
  - $y_{i,t}$ 为 $y_i$ 的第t个token，$y_{i,<t}$ 为 $y_i$ 的第1,2,...,t-1个token
  - $A_{y_{i,t}}$ 为token $y_{i,t}$ 的优势
  - $π_θ$ 是训练框架（例如fsqp）部署的LLM，$π_{θ.detach}$ 是训练框架（例如vllm）部署的LLM
  - 所有response $y_i$ 是从 $π_{θ.detach}$ rollout出来的

## 目标函数：

```math
\begin{aligned}
J(θ) 
&= \mathbb{E}_{x \sim D(x), y \sim π_θ(y|x)_{.detach}} \left[ \sum_{t=1}^{|y|} A_{y_t} \cdot \frac{π_θ(y_t|x, y_{<t})}{π_θ(y_t|x, y_{<t})_{.detach}} \right] \\
&≈ \frac{1}{N} \sum_{采样 \atop N个x} \frac{1}{G} \sum_{每个x \atop 生成G个{y_i}} \left[ \sum_{t=1}^{|y|} A_{y_{i,t}} \cdot \frac{π_θ(y_{i,t}|x, y_{i,<t})}{π_θ(y_{i,t}|x, y_{i,<t})_{.detach}} \right]
\end{aligned}
```
其中单个token $y_{i,t}$ 的优势：

$$
A_{y_{i,t}} = \frac{1}{|y_i|} \cdot \frac{r_i - \text{mean}\left(\{r_i\}_{i=i}^G\right)}{\text{std}\left(\{r_i\}_{i=i}^G\right)} \qquad 与t无关
$$

## 求导

```math
\begin{aligned}
\nabla_θ J(θ) 
&= \mathbb{E}_{x \sim D(x), y \sim π_θ(y|x)_{.detach}} \left[ \sum_{t=1}^{|y|} A_{y_t} \cdot \frac{π_θ(y_t|x, y_{<t})}{π_θ(y_t|x, y_{<t})_{.detach}} \nabla_θ \log π_θ(y_t|x, y_{<t})  \right] \\
&≈ \frac{1}{N} \sum_{采样 \atop N个x} \frac{1}{G} \sum_{每个x \atop 生成G个{y_i}} \left[ \sum_{t=1}^{|y|} A_{y_{i,t}} \cdot \frac{π_θ(y_{i,t}|x, y_{i,<t})}{π_θ(y_{i,t}|x, y_{i,<t})_{.detach}} \nabla_θ \log π_θ(y_t|x, y_{<t}) \right]
\end{aligned}
```

## 讨论
以上是GRPO公式**最本质**的部分。完整的GRPO只是加上了一些trick：
- $π_{old}$ ：为了把on-policy变成off-policy以节约rollout成本，需要使用重要性采样
- clip操作：off-policy情况下为了防止 $π_{old}$ 和 $π_θ$ 差距过大，导致训练不稳定
- KL散度：为了防止RL训练后的 $π_θ$ 和其最初始值 $π_{ref}$ 偏离太远，导致灾难性遗忘

$\frac{1}{|y_i|}$ 在公式中的位置？
- GRPO公式的常见写法中，$\frac{1}{|y_i|}$ 不出现在优势 $A$ 里，而出现在 $\frac{1}{|y_i|} \sum_{i=1}^{|y_i|}$ 位置
- 两种写法是等价的，我的写法更本质： $\frac{1}{|y_i|}$ 来自于严格推导出的单个token的优势

公式中 $\frac{π_θ}{π_{θ.detach}}$ 为什么不移除？
- 其反映了RL实质的训练过程：从 $π_{θ.detach}$ rollout，从 $π_θ$ 更新参数，两者采用不同的部署框架（而且数值经常有差异，训练-推理不一致是RL中一个头疼的问题）
- 即使二者数值相等也不能移除，它们在计算图上有实际对应，移除后无法对 $π_θ$ 求导

on-policy版本的GRPO，目标函数是0吗？
- 因为 $A$ 是通过reward归一化得到的。如果不考虑 $π_{θ.detach}$ 和 $π_θ$ 的数值差异，$J(θ)$ 确实等于0。但梯度 $J(θ)≠0$ 。后面章节会详细介绍为什么梯度不是0。


# 拓展：PPO和GSPO

有了以上GRPO最本质的公式，可以很容易拓展出PPO和GSPO的公式

## PPO：用网络估计优势A

PPO中，第t个token的优势 $A_t$ 不是通过数据归一化得到的，而是用神经网络计算的：  

$$
A_t = γ^{|y|-t} \cdot r - V_θ(s_t = x+y_{<t})
$$

- $r$ 是最终response
- $V_θ(s_t = x+y_{<t})$ 是一个估计状态 $s_t = x+y_{<t}$ 的神经网络。
 
价值网络 $V_θ(x+y_{<t})$ 的形式，与LLM计算句子概率是类似的： $π_θ(x+y_{<t})$ ，因此在实现上可以和 $π_θ(x+y_{<t})$ 共用相同的LLM，只是在输出时换用不同的head。

> 以上只是PPO最核心的思想。更多细节就不展开了（例如V如何训练、用广义优势估计GAE计算A）

PPO与GRPO比较：
- GRPO的A直接统计得到，本质是蒙特卡洛；PPO的A用网络估计，本质是时序差分（时序差分是RL中另一大类方法，这里不做探讨）
- PPO和GRPO的训练效果：PPO低方差、高偏差；GRPO高方差、低偏差
  - 如何理解？
  - PPO低方差：用神经网络估计A，本质是用大量数据的滑动平均，波动比GRPO每次采样小
  - PPO高偏差：（1）网络不一定有能力拟合真实的A；（2）训练过程中，网络可能还未收敛

## GSPO：把整个response作为一个大动作

如果把整个response $y_i$ 视为一个完整动作，得到的就是GSPO。

目标函数：

```math
\begin{aligned}
J(θ) 
&= \mathbb{E}_{x \sim D(x), y \sim π_θ(y|x)_{.detach}} \left[ A_{y} \cdot \frac{π_θ(y|x)}{π_θ(y|x)_{.detach}} \right] \\
&≈ \frac{1}{N} \sum_{采样 \atop N个x} \frac{1}{G} \sum_{每个x \atop 生成G个{y_i}} \left[  A_{y_{i}} \cdot \frac{π_θ(y_{i}|x)}{π_θ(y_{i}|x)_{.detach}} \right]
\end{aligned}
```
其中完整response $y_{i}$ 的优势：

$$
A_{y_{i}} = \frac{r_i - \text{mean}\left(\{r_i\}_{i=i}^G\right)}{\text{std}\left(\{r_i\}_{i=i}^G\right)}
$$

它去掉了GRPO中每个token优势相等的假设。

> 注：
> - on-policy下GSPO和GRPO是完全等价的，因为 $\frac{π_θ}{π_{θ.detach}}$ 都是1.   
> - off-policy下把 $π_{θ.detach}$ 换成 $π_{old}$ ，两者就不相等了。

# 重要性采样

前面已经介绍过重要性采样，而本章将进行更深入的探讨。

基于重要性采样的公式： $\mathbb{E}_{x \sim p(x)} [f(x)] = \mathbb{E}_{x \sim q(x)} \left[ \frac{p(x)}{q(x)} f(x) \right]$ ，我们可以从任意一个分布中rollout数据，来训练GRPO：

$$
\mathbb{E}_{x \sim D(x), y \sim π_θ(y|x)} [G(y)] = \mathbb{E}_{x \sim D(x), y \sim q(y|x)} \left[ \frac{π_θ(y|x)}{q(y|x)} G(y) \right]
$$

- $π_θ(y|x)$ 是待训练的LLM
- $q(y|x)$ 是用来rollout的数据分布（通常也是个LLM）

当我们把用于rollout的 $q(y|x)$ 换成不同的分布，可以统一看待 on-policy GRPO、off-policy GRPO、LLM知识蒸馏，甚至SFT。

## on-policy：用 $π_{θ.detach}$ rollout

前面推导GRPO的公式，用的就是on-policy。其中用于rollout的 $π_{θ.detach}$ ，对应了用vllm等推理专用框架部署的LLM，其参数值和 $π_{θ}$ 相同。这些内容已经详细探讨过了。

这里重点解释一个问题：on-policy GRPO中，为什么目标函数（优势之和）值是0，但梯度不是0。

- 计算图的角度：虽然数值是0，但计算图中有 $π_θ(y|x)$ 节点，保证它会有梯度
- 动作的角度：把优势视为一种“效应”，虽然所有动作的“效应”之和是0，但单个动作的“效应”都不是0。而且这些效应更新的方向不是均衡的：好的动作会被加强，坏的动作会被削弱
- 数值的角度（随着训练动态更新）：当前优势的期望之和恰好是0，但策略更新后，如果恰好还是rollout出这些数据，在旧的策略视角下，期望之和就不是0了（虽然在新策略的视角下还是0）

response空间无限的例子不容易理解，这里举一个有限的例子
  - prompt：选择题，ABCD四选一，其中正确选项是C
  - response：ABCD四个token其中之一
  - 采样：采样20个response $\{y_1, ..., y_{20}\}$ ，每个选项一定会重复多次。选项的分布就是策略 $π_θ$
  - 对20个response归一化，得到 $μ$ 和 $σ$ ，计算A，它们的和确实是0
  - 梯度的方向：加强C。削弱ABD
  - 参数更新后，会得到一个新策略 $π_θ^*$ ，采样20个新response $\{y'_1, ..., y'_{20}\}$
  - 结果： $\{y'_1, ..., y'_{20}\}$ 中的正确选项C比 $\{y_1, ..., y_{20}\}$ 多
  - 对20个新response归一化，得到 $μ'$ 和 $σ'$ ，计算A'，它们的和确实还是0
    - 但A'之和为0是在 新策略 $π_θ^*$ 视角下的
    - 在旧策略 $π_θ$ 视角下如果计算 $\{y'_1, ..., y'_{20}\}$ 的优势，应当使用旧的 $μ$ 和 $σ$ ，而不是新的 $μ'$ 和 $σ'$ ，这样算出的优势就不是0了
  - 也就是说，优势和为0只是“临时”的，如果我们永远从固定策略中rollout，下个时刻优势和就不是0了
  - 之所以每次目标函数的优势和为0，是因为on-policy每次都重新换策略
  - 因此on-policy中目标函数一直是0，但梯度不是0


## off-policy: 用模型的历史版本 $π_{old}$ rollout

GRPO公式中，更常见的是off-policy形式：  


```math
\begin{aligned}
J(θ) 
&= \mathbb{E}_{x \sim D(x), y \sim π_{old}(y|x)} \left[ \sum_{t=1}^{|y|} A_{y_t} \cdot \frac{π_θ(y_t|x, y_{<t})}{π_{old}(y_t|x, y_{<t})} \right] \\
&≈ \frac{1}{N} \sum_{采样 \atop N个x} \frac{1}{G} \sum_{每个x \atop 生成G个{y_i}} \left[ \sum_{t=1}^{|y|} A_{y_{i,t}} \cdot \frac{π_θ(y_{i,t}|x, y_{i,<t})}{π_{old}(y_{i,t}|x, y_{i,<t})} \right]
\end{aligned}
```

它实际对应了这样的训练过程：
- 在某一时刻，用于训练的 $π_θ$ 和用于推断的 $π_{old}$ 参数相同（但部署框架不同）
  - 用推理模型 $π_{old}$ rollout出 M*N 条数据
  - 对于训练模型 $π_θ$ ，每次取 M 条数据用于更新梯度，一共更新 N 轮
  - 最初始 $π_θ = π_{old}$ ，但参数 $π_θ$ 每轮都在更新，而 $π_{old}$ 一直保持不变，且两者相差越来越大
  - 第N轮更新完后，使用 $π_θ$ 参数重新部署用于rollout的 $π_{old}$
- 此时两者参数又相等了，开始下个 N 轮的训练

训练时为什么不把 M*N 条数据一次全用掉？
- 显存问题：推理框架（如vllm）比训练框架（如fsdp）显存占用少，如果rollout M\*N条数据能把显存跑满，训练用M\*N条数据一定会爆显存

为什么不通过梯度累加，用M\*N条数据更新一次参数？
- 没有必要，更新N次 vs 更新1次，前者更新频率更高，训练更快（但方差也更大）
- 使用M条数据更新时，本身往往也用了梯度累加（即真实的batch size比M更小，现在的框架如verl可以自动分配实际batch size并自动累加到M）

off-policy相比于on-policy是一种trade-off，能提升训练效率，代价是方差变大，训练更不稳定。
> 某些教程中写off-policy是为了rollout一次训练多次，这个说法有些误导：并不是rollout出M条数据，每轮更新都用这M条，一共更新N次；而是一次rollout出M*N条数据，每次用M条，一共分N次用完

### verl中对应的训练config（暂时不管evaluate）：
- `data.train_batch_size`：一次选择多少个prompt来rollout
- `actor_rollout_ref.rollout.n`: 一个prompt rollout出多少个response
- `actor_rollout_ref.actor.ppo_mini_batch_size`： $π_θ$ 更新一次使用多少个prompt（以及相应的全部response）
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`：在gpu上计算时实际的batch size（会通过梯度累加实现ppo_mini_batch_size）
- `actor_rollout_ref.actor.use_dynamic_bsz`: 是否自动、动态设置ppo_micro_batch_size_per_gpu

情况举例：
- vllm一次选择 `train_batch_size` 个prompt，每个rollout出 `n` 个response，一共 `train_batch_size × n` 个response 
- fsdp一次选择 `ppo_mini_batch_size` 个prompt（相应`ppo_mini_batch_size × n`个response）用于更新模型，一共更新 `train_batch_size / ppo_mini_batch_size` 轮
- 如果未设置`use_dynamic_bsz`，则gpu实际使用`ppo_micro_batch_size_per_gpu`个prompt，并且累加到`ppo_mini_batch_size`完成一次梯度更新，一共累加 `ppo_mini_batch_size / ppo_micro_batch_size_per_gpu` 轮

## LLM知识蒸馏：用另一个更强大的LLM rollout

如果我们的 $π_{θ}(y|x)$ 是个弱LLM，而采样分布 $q(y|x)$ 是个强LLM，且我们始终从强LLM rollout并实时计算 $q(y|x)$ ，则对应了LLM知识蒸馏：用强模型生成的数据训练弱模型。

这也是种极端版本的off-policy：用于采样的模型永远不变。

> 这只是种理论做法。实际的知识蒸馏，通常是用强LLM rollout出大量数据，给弱模型SFT。

## SFT：从数据集的（prompt, response）分布rollout

SFT其实也可以套到策略梯度+重要性采样的框架下。只不过场景特殊：

- rollout采样过程：
  - 采样数据 $x \sim D(x)$ ：从数据集采样 prompt $x$
  - rollout $y \sim q(y|x)$ ：这样定义 $q(y,x)$ ，则采样出的 $y$ 有且仅有数据集中的gold response。
  - 此结果等价于采样 $(x,y) \sim D(x,y)$
```math
q(y|x) = 
\begin{cases}
1  \qquad \text{如果y是数据集中的gold response} \\
0 \qquad \text{otherwise.}
\end{cases}
```

- RL建模：
  - token级建模
  - reward粒度：token级，生成gold token则为1，否则为0
  - 折扣因子γ=0，即只看当前token reward
  - 轨迹片段收益 $G_t = \sum_{i=0}^{|y|} γ^i r_{t+i} = r_t$
    - 但因为只会rollout出gold response，所以实际上只有 $G_t=r_t=1$

- 带入：
  - $G_t$ 版本的导数（因此实际是REINFORCE而不是GRPO）：

```math
\begin{aligned}
\nabla_θ J_θ
&= \nabla_θ \mathbb{E}_{x \sim D(x), y \sim q(y|x, y)} \left[ \sum_{t=1}^{|y|} \frac{π_θ(y_t|x， y_{<t})}{q(y_t|x, y_{<t})} \cdot G_t \right] \\
&= \mathbb{E}_{(x,y) \sim D(x,y)} \left[ \sum_{t=1}^{|y|} \frac{\nabla_θ  π_θ(y_t|x， y_{<t})}{1} \cdot 1 \right] \qquad 因为只会采样出 \text{gold} \ y \\
\end{aligned}
```

可以发现该结果就是SFT监督学习的公式。