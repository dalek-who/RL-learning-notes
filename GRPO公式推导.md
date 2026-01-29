<h1 align="center">RL基础知识 & GRPO公式推导</h1>

# RL基础概念：S、A、R、π、环境

## 状态s、动作a、奖励r、策略π

在用强化学习训练LLM的语境中，有两种建模级别。
- token级建模：每个生成的token视为一个action（对应PPO）
- sequence级建模：每个完整生成序列视为一个action（对应GRPO）

相应的，把强化学习的概念对应到LLM上：
**token级建模**
- 状态s：
  - 初始状态$s_0$：prompt $x$
  - t时刻状态$s_t$：prompt $x$ + response的前t个token $y_{1:t}$
  - （*此处 $y_i$ 指的是序列 y 的第 i 个token*）
- 动作a：
  - t时刻的动作$a_t$：response的第t个token $y_{t}$
- 奖励r：
  - 最后一个token：完整序列的奖励（例如答案是否正确）
  - 前面的token：奖励为0
- 策略π：
  - LLM生成第t个token的概率 $P(y_t | x, y_{1:t-1})$

**sequence级建模（只有一问一答）**
- 状态s：只有初始、结束两个状态
  - 初始状态$s_0$：prompt $x$
  - 结束状态$s_1$：prompt $x$ + response $y$
- 动作a：只有一步动作
  - LLM生成的response $y$
- 奖励r：
  - 完整序列的奖励（例如答案是否正确）
- 策略π：
  - LLM生成response y的概率 $P(y | x)$

**sequence级建模（多轮问答）**：常见于多轮agent的RL训练，但也可以全拆成单轮
- 状态s：
  - 初始状态$s_0$：初始prompt $x_{1}$
  - t时刻状态$s_t$：初始prompt $x_{1}$ + 第1轮response $y_{1}$ + ... + 第t轮prompt $x_{t}$ + 第t轮response $y_{t}$
  - （*此处的 $x_i,y_i$ 指代的是第 i 个 x,y 序列，而不是序列中第i个token*）
- 动作a：
  - t时刻的动作$a_t$：第t轮response $y_{t}$
- 奖励r：
  - t时刻的奖励$r_t$：第t轮回复的奖励（有些方法有中间奖励），如果t是最后一轮则对应终局奖励
- 策略π：
  - LLM生成第t个response的概率 $P(y_t | x_1, y_1, ..., x_{t-1}, y_{t-1}, x_t)$

## 环境：确定性环境 & 概率环境

概率$P(s',r|s,a)$
todo

```
状态（State）是智能体在某个时刻对环境的完整描述。在马尔可夫决策过程（MDP）中，状态满足马尔可夫性质：未来只依赖于当前状态，与历史无关。

- **数学表示**：$s \in \mathcal{S}$，其中 $\mathcal{S}$ 是状态空间
- **性质**：$P(s_{t+1} | s_t, a_t, s_{t-1}, ...) = P(s_{t+1} | s_t, a_t)$

## 动作 a

动作（Action）是智能体在给定状态下可以执行的操作。动作的选择会影响环境的状态转移和获得的奖励。

- **数学表示**：$a \in \mathcal{A}(s)$，其中 $\mathcal{A}(s)$ 是状态 $s$ 下的动作空间
- **动作空间类型**：
  - 离散动作空间：$\mathcal{A} = \{a_1, a_2, ..., a_n\}$
  - 连续动作空间：$\mathcal{A} \subseteq \mathbb{R}^n$

## 奖励 r

奖励（Reward）是环境对智能体执行动作后的即时反馈信号。奖励函数定义了任务的目标。

- **数学表示**：$r_t = R(s_t, a_t, s_{t+1})$ 或 $r_t = R(s_t, a_t)$
- **性质**：奖励是标量值，可以是正数（鼓励）或负数（惩罚）
- **目标**：最大化累积奖励（回报）

## 策略 π

策略（Policy）定义了智能体在给定状态下选择动作的概率分布。策略是强化学习的核心，决定了智能体的行为。

- **数学表示**：
  - 随机策略：$\pi(a|s) = P(a|s)$，表示在状态 $s$ 下选择动作 $a$ 的概率
  - 确定性策略：$\pi(s) = a$，直接映射状态到动作
- **性质**：
  - $\sum_{a \in \mathcal{A}} \pi(a|s) = 1$（概率归一化）
  - $\pi(a|s) \geq 0$（非负性）

## 确定性环境 & 概率环境

### 确定性环境

在确定性环境中，给定状态和动作，下一个状态是唯一确定的：

$$s_{t+1} = f(s_t, a_t)$$

- 状态转移函数是确定性的
- 奖励函数通常是确定性的：$r_t = R(s_t, a_t)$
- 简化了分析和计算

### 概率环境

在概率环境（随机环境）中，状态转移和奖励都是随机的：

$$P(s_{t+1}|s_t, a_t) \text{ 是概率分布}$$

$$P(r_t|s_t, a_t) \text{ 是概率分布}$$

- 更符合现实世界的复杂性
- 需要处理不确定性

### 举例：步枪打靶
todo

## 在LLM中的对应

在大型语言模型（LLM）的强化学习场景中：

- **状态 s**：当前生成的文本序列（token序列），即 $s = [x_1, x_2, ..., x_t]$
- **动作 a**：下一个要生成的token，即 $a = x_{t+1}$
- **奖励 r**：对生成文本的质量评估（如人类反馈、奖励模型评分等）
- **策略 π**：语言模型的生成策略，即 $P(x_{t+1}|x_1, ..., x_t)$，由模型参数 $\theta$ 参数化
- **环境**：通常是确定性的（给定prompt和生成策略，输出是确定的），但在训练过程中策略会变化

todo：两种对应方式：动作是token vs 动作是序列

# 衍生概念：G、V、A、Q

## 回报G

回报（Return）是从某个时刻开始到episode结束的累积奖励，是智能体的长期目标。

- **数学表示**：
  $$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$
  
  其中 $\gamma \in [0, 1]$ 是折扣因子（discount factor）

- **有限horizon情况**：
  $$G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1}$$

- **无折扣情况**（$\gamma = 1$）：
  $$G_t = \sum_{k=0}^{T-t-1} r_{t+k+1}$$

## 价值V

状态价值函数（State Value Function）$V^{\pi}(s)$ 表示在策略 $\pi$ 下，从状态 $s$ 开始的期望回报。

- **数学定义**：
  $$V^{\pi}(s) = \mathbb{E}_{\pi}[G_t | s_t = s] = \mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \Big| s_t = s\right]$$

- **含义**：衡量在策略 $\pi$ 下，状态 $s$ 的"好坏程度"

## 优势A

优势函数（Advantage Function）$A^{\pi}(s, a)$ 表示在状态 $s$ 下执行动作 $a$ 相对于平均水平的优势。

- **数学定义**：
  $$A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)$$

- **含义**：
  - $A^{\pi}(s, a) > 0$：动作 $a$ 优于平均水平
  - $A^{\pi}(s, a) < 0$：动作 $a$ 劣于平均水平
  - $A^{\pi}(s, a) = 0$：动作 $a$ 等于平均水平

## Q函数

动作价值函数（Action-Value Function）$Q^{\pi}(s, a)$ 表示在策略 $\pi$ 下，在状态 $s$ 执行动作 $a$ 后的期望回报。

- **数学定义**：
  $$Q^{\pi}(s, a) = \mathbb{E}_{\pi}[G_t | s_t = s, a_t = a] = \mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \Big| s_t = s, a_t = a\right]$$

- **含义**：衡量在策略 $\pi$ 下，状态-动作对 $(s, a)$ 的"好坏程度"

- **与V函数的关系**：
  $$V^{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a|s) Q^{\pi}(s, a) = \mathbb{E}_{a \sim \pi(\cdot|s)}[Q^{\pi}(s, a)]$$

## 贝尔曼方程 & R/V/Q/A之间的关系

### 标准意义下（概率环境）

在概率环境中，贝尔曼方程描述了价值函数之间的递归关系：

**V函数的贝尔曼方程**：
$$V^{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{s' \in \mathcal{S}} P(s'|s, a) \left[R(s, a, s') + \gamma V^{\pi}(s')\right]$$

**Q函数的贝尔曼方程**：
$$Q^{\pi}(s, a) = \sum_{s' \in \mathcal{S}} P(s'|s, a) \left[R(s, a, s') + \gamma \sum_{a' \in \mathcal{A}} \pi(a'|s') Q^{\pi}(s', a')\right]$$

**关系总结**：
- $Q^{\pi}(s, a) = \mathbb{E}_{s' \sim P(\cdot|s,a)}[R(s, a, s') + \gamma V^{\pi}(s')]$
- $V^{\pi}(s) = \mathbb{E}_{a \sim \pi(\cdot|s)}[Q^{\pi}(s, a)]$
- $A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)$

### 确定性环境中的简化形式

在确定性环境中，状态转移是确定的，贝尔曼方程简化为：

**V函数的贝尔曼方程**：
$$V^{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \left[R(s, a) + \gamma V^{\pi}(f(s, a))\right]$$

其中 $s' = f(s, a)$ 是确定性的状态转移函数。

**Q函数的贝尔曼方程**：
$$Q^{\pi}(s, a) = R(s, a) + \gamma \sum_{a' \in \mathcal{A}} \pi(a'|s') Q^{\pi}(s', a')$$

其中 $s' = f(s, a)$。

## 在LLM中的对应

在LLM的强化学习场景中：

- **回报G**：从当前位置到序列结束的累积奖励
  $$G_t = \sum_{k=t}^{T} r_k$$
  
  其中 $r_k$ 是对第 $k$ 个token或整个序列的奖励

- **价值V**：$V^{\pi}(s)$ 表示从当前文本序列 $s$ 开始的期望回报
  $$V^{\pi}([x_1, ..., x_t]) = \mathbb{E}_{\pi}[G_t | \text{当前序列}]$$

- **Q函数**：$Q^{\pi}(s, a)$ 表示在当前序列 $s$ 下生成token $a$ 后的期望回报
  $$Q^{\pi}([x_1, ..., x_t], x_{t+1}) = \mathbb{E}_{\pi}[G_t | \text{当前序列}, \text{下一个token}]$$

- **优势A**：$A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)$，衡量生成某个token相对于平均水平的优势

# 蒙特卡洛：估计V/A/Q

（时序差分最终演化为PPO，蒙特卡洛最终演化为GRPO，理解GRPO并不需要理解时序差分）

## 蒙特卡洛方法的基本思想

蒙特卡洛方法通过采样完整的episode来估计价值函数，不需要知道环境模型（model-free）。

### 估计V函数

通过采样多个episode，计算从状态 $s$ 开始的平均回报：

$$\hat{V}^{\pi}(s) = \frac{1}{N} \sum_{i=1}^{N} G_t^{(i)}$$

其中 $G_t^{(i)}$ 是第 $i$ 个episode中从状态 $s$ 开始的回报。

### 估计Q函数

类似地，通过采样估计Q函数：

$$\hat{Q}^{\pi}(s, a) = \frac{1}{N} \sum_{i=1}^{N} G_t^{(i)}$$

其中 $G_t^{(i)}$ 是第 $i$ 个episode中从状态-动作对 $(s, a)$ 开始的回报。

### 估计优势函数

优势函数可以通过Q函数和V函数计算：

$$\hat{A}^{\pi}(s, a) = \hat{Q}^{\pi}(s, a) - \hat{V}^{\pi}(s)$$

或者直接通过采样计算：

$$\hat{A}^{\pi}(s, a) = G_t - \hat{V}^{\pi}(s)$$

其中 $G_t$ 是从 $(s, a)$ 开始的回报，$\hat{V}^{\pi}(s)$ 是从状态 $s$ 的平均回报。

## 蒙特卡洛策略梯度

策略梯度方法通过直接优化策略参数来最大化期望回报。策略梯度定理给出：

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \cdot G_t\right]$$

其中 $J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}[G_0]$ 是期望回报。

### 使用优势函数

引入优势函数可以减少方差：

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \cdot A^{\pi}(s_t, a_t)\right]$$

因为优势函数 $A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)$ 的期望为0，所以不会改变梯度，但可以减少方差。

## 在LLM中的对应

在LLM场景中，蒙特卡洛方法的应用：

### 采样完整序列

1. **生成完整序列**：给定prompt，使用当前策略 $\pi_{\theta}$ 生成多个完整序列
2. **计算回报**：对每个序列计算奖励 $r$（可能只在序列结束时给出）
3. **估计价值**：通过多个样本的平均值估计价值函数

### 简化场景：确定性环境 + 单步决策

在GRPO的简化场景中：
- **确定性环境**：给定prompt和策略，生成是确定的（但在训练过程中策略会变化）
- **单步决策**：通常考虑的是生成下一个token的决策
- **回报计算**：可能基于完整序列的奖励，或者基于当前token的即时奖励

### GRPO中的蒙特卡洛估计

在GRPO中，通过以下方式估计优势：

1. **采样多个序列**：对同一prompt，生成多个完整序列
2. **计算回报**：$G_t = r$（序列级别的奖励）
3. **估计优势**：通过组内比较（group relative）来估计优势，而不是直接估计V函数

这种方法避免了需要单独估计V函数，而是通过组内相对比较来估计优势，这正是GRPO的核心思想。

# GRPO：简化场景中的蒙特卡洛

GRPO = 确定性环境 + 单步决策下的蒙特卡洛

## GRPO的核心思想

GRPO（Group Relative Policy Optimization）是一种在简化场景下的蒙特卡洛策略优化方法，特别适用于LLM的强化学习训练。

### 简化假设

1. **确定性环境**：给定prompt和策略，生成过程是确定的
2. **单步决策**：关注的是生成下一个token的决策
3. **组内比较**：通过同一prompt下的多个生成样本进行相对比较

## GRPO的数学推导

### 标准策略梯度

标准的策略梯度方法使用优势函数：

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{s \sim \rho^{\pi}, a \sim \pi_{\theta}} \left[\nabla_{\theta} \log \pi_{\theta}(a|s) \cdot A^{\pi}(s, a)\right]$$

其中 $\rho^{\pi}$ 是状态分布。

### GRPO的优势估计

在GRPO中，优势函数通过组内相对比较来估计：

对于同一prompt $s$，生成 $K$ 个完整序列 $\{\tau_1, \tau_2, ..., \tau_K\}$，每个序列的奖励为 $\{r_1, r_2, ..., r_K\}$。

**组内平均奖励**：
$$\bar{r} = \frac{1}{K} \sum_{i=1}^{K} r_i$$

**优势估计**：
对于序列 $\tau_i$ 中的每个状态-动作对 $(s_t, a_t)$，优势估计为：

$$\hat{A}(s_t, a_t) = r_i - \bar{r}$$

这相当于用组内平均作为基线（baseline），避免了需要单独估计V函数。

### GRPO的目标函数

GRPO的目标函数可以写为：

$$L^{GRPO}(\theta) = \mathbb{E}_{s \sim \mathcal{D}} \left[\mathbb{E}_{\{\tau_i\}_{i=1}^{K} \sim \pi_{\theta}} \left[\frac{1}{K} \sum_{i=1}^{K} \sum_{t=0}^{T_i} \log \pi_{\theta}(a_t^{(i)}|s_t^{(i)}) \cdot (r_i - \bar{r})\right]\right]$$

其中：
- $\mathcal{D}$ 是prompt分布
- $K$ 是每个prompt生成的序列数量
- $T_i$ 是第 $i$ 个序列的长度
- $r_i$ 是第 $i$ 个序列的奖励
- $\bar{r} = \frac{1}{K} \sum_{j=1}^{K} r_j$ 是组内平均奖励

### 梯度计算

对参数 $\theta$ 求梯度：

$$\nabla_{\theta} L^{GRPO}(\theta) = \mathbb{E}_{s \sim \mathcal{D}} \left[\mathbb{E}_{\{\tau_i\}_{i=1}^{K} \sim \pi_{\theta}} \left[\frac{1}{K} \sum_{i=1}^{K} \sum_{t=0}^{T_i} \nabla_{\theta} \log \pi_{\theta}(a_t^{(i)}|s_t^{(i)}) \cdot (r_i - \bar{r})\right]\right]$$

### 关键特性

1. **无偏性**：由于 $\mathbb{E}[\bar{r}] = \mathbb{E}[r_i]$，使用组内平均作为基线不会引入偏差
2. **方差减少**：组内比较可以减少估计的方差
3. **无需价值函数**：不需要单独训练价值函数，简化了训练过程
4. **适合LLM**：特别适合LLM场景，因为可以自然地生成多个候选序列

## GRPO vs PPO

| 特性 | GRPO | PPO |
|------|------|-----|
| 价值估计方法 | 蒙特卡洛（组内比较） | 时序差分（价值函数） |
| 基线 | 组内平均奖励 | 价值函数 $V(s)$ |
| 优势估计 | $\hat{A} = r_i - \bar{r}$ | $\hat{A} = r + \gamma V(s') - V(s)$ |
| 适用场景 | 确定性环境，序列级奖励 | 一般MDP，即时奖励 |
| 训练复杂度 | 较低（无需价值网络） | 较高（需要价值网络） |

## 总结

GRPO通过以下方式简化了强化学习训练：

1. **确定性环境假设**：简化了状态转移
2. **组内相对比较**：用组内平均作为基线，避免估计V函数
3. **蒙特卡洛方法**：通过完整序列采样估计优势
4. **适合LLM**：天然适合生成多个候选序列的场景

这使得GRPO成为LLM强化学习训练的一个简洁而有效的方法。
```