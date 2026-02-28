# KL散度

> 可参考这个教程：
> <https://xihuai18.github.io/reinforcement-learning/2025/12/01/kl-estimators-zh.html>

GRPO中KL散度的作用：防止和初始化的参数 $π_{ref}$ 相差太远，导致灾难性遗忘（ $π_{ref}$ 通常是SFT训练好的模型，如果和 $π_{ref}$ 相差过远，可能会丢失LLM的通用能力，甚至生成乱码）
> 但实践中往往不使用KL散度，甚至后续的DAPO、GSPO明确去掉了KL散度

KL散度在机器学习中，通常用于衡量两个分布的差异。但我们对KL散度的理解往往不够深刻。此章将跟深入KL散度的本质，并介绍在GRPO（乃至各种强化学习中）使用KL散度的注意事项。

同样先给结论：在GRPO中使用KL散度要注意一些细节
- KL散度的“方向”：是 $\text{KL}(π_θ | π_{ref})$ ，而不是$\text{KL}(π_{ref} | π_θ)$ 
- KL散度出现的位置：出现在目标函数的正则项上，而不是reward上
- KL散度的实际计算方式：使用 $k_3$ 估计，而不是原始定义 $k_1$


## KL散度的“方向”

**KL散度的不对称性**： $\text{KL}(π_{ref} | π_θ) ≠ \text{KL}(π_θ | π_{ref})$
- 正向KL散度： $\text{KL}(π_{ref} | π_θ) = \mathbb{E}_{τ \sim π_{ref}} \left[ \log \frac{π_{ref}}{π_θ} \right] ≈ \sum_{从π_{ref} \atop 采样τ} π_{ref} \cdot \log \frac{π_{ref}}{π_θ}$
  - 使用场景：知识蒸馏、SFT（极小化正向KL散度）
  - 知识蒸馏： $π_{ref}$ 是教师模型的分布（或教师模型rollout出的SFT数据集）
  - SFT： $π_{ref}$ 是数据集（极小化交叉熵等价于极小化正向KL散度）
  
- **逆向KL散度**： $\text{KL}(π_θ | π_{ref}) = \mathbb{E}_{τ \sim π_θ} \left[ \log \frac{π_θ}{π_{ref}} \right] ≈ \sum_{从π_θ \atop 采样τ} π_θ \cdot \log \frac{π_θ}{π_{ref}}$
  - 使用场景：**强化学习中（例如GRPO）通常都使用逆向KL散度**

两种KL散度的性质差异：
- 假设 $π_{ref}$ 的分布有少数“高峰”+大量“矮峰” （矮峰也是峰，而不是谷）
  - 极小化 $\text{KL}(π_{ref} | π_θ)$ ： $π_θ$ 通常会变得宽而扁，即倾向于囊括 $π_{ref}$ 的所有高峰、矮峰，但高峰可能没有 $π_θ$ 高 
  - 极小化 $\text{KL}(π_θ | π_{ref})$ ：$π_θ$ 通常会变得窄而尖，即倾向于拟合 $π_{ref}$ 的高峰，忽略矮峰
- 直观（但不严格）的理解：
  - $\text{KL}(π_{ref} | π_θ)$ ：样本从 $π_{ref}$ 中采样，无论高峰矮峰都会采到。由于 $π_{ref}$ 在训练中不变，矮峰的数据被采到的概率不会降低。因此 $π_{θ}$ 需要兼顾高、矮峰的概率都不能太低，最终变得宽而扁，囊括 $π_{ref}$ 的所有高峰、矮峰
  - $\text{KL}(π_θ | π_{ref})$ ：样本从 $π_θ$ 中采样，而训练中 $π_θ$ 会不断变化，高峰越来越高，被采样的概率加大，而矮峰被采样的概率减小。因此最终会变得窄而尖，拟合 $π_{ref}$ 的高峰，而忽略矮峰
- 基于以上性质的应用：
  - SFT：借助 $\text{KL}(π_{ref} | π_θ)$ “宽”的优点，来提升模型的通用能力，但可能博而不精（高峰够不高）
  - GRPO：借助 $\text{KL}(π_θ | π_{ref})$ “高”的优点，来提升模型的专项能力，但可能遗忘通用能力（只剩几个高峰，低峰都没了）
- GRPO不使用 $\text{KL}(π_{ref} | π_θ)$ 的原因：
  - 可能导致约束过强，强化学习更新不动。
  - 本质上等于永远从初始化时的 $π_{ref}$ 来rollout（虽然实际是从 $π_{old}$ rollout，但经过重要性采样修正后，本质上也等价于从 $π_{ref}$ rollout），这样即使 $π_θ$ 已经改进了，新的数据可能rollout不出来。
  - 类似于永远在修正已经不会再犯的旧错误


## KL散度的求导
推导过程太麻烦，略过了，直接给结论。

- 正向KL散度（SFT）
  - forward： $\text{KL}(π_{ref} | π_θ) = \mathbb{E}_{τ \sim π_{ref}} \left[ \log \frac{π_{ref}}{π_θ} \right]$
  - backward：  $\nabla_θ \text{KL}(π_{ref} | π_θ) = \mathbb{E}_{τ \sim π_{ref}} \left[ - \nabla_θ \log π_θ \right]$
  - 其中 $\mathbb{E}_{τ \sim π_{ref}} [\cdot] ≈ \frac{1}{N} \sum_{从 π_{ref} 采样 \atop N个τ} (\cdot)$

- 逆向KL散度（GRPO）
  - forward： $\text{KL}(π_θ | π_{ref}) = \mathbb{E}_{τ \sim π_θ} \left[ \log \frac{π_θ}{π_{ref}} \right]$
  - backward： $\nabla_θ \text{KL}(π_θ | π_{ref}) = \mathbb{E}_{τ \sim π_{θ.detach}} \left[ \frac{π_{θ}}{π_{θ.detach}} \log \frac{π_θ}{π_{ref}} \nabla_θ \log π_θ \right]$
  - 其中 $\mathbb{E}_{τ \sim π_{θ.detach}} [\cdot] ≈ \frac{1}{N} \sum_{从 π_{θ.detach} 采样 \atop N个τ} (\cdot)$

特殊情况：一条样本的KL散度
- 特殊之处：在 $\mathbb{E}≈\frac{1}{N} \sum_{N个采样}$ 的转换中 $N=1$
- 正向KL散度：
  - forward： $\text{KL}(π_{ref} | π_θ) = \mathbb{E}_{τ \sim π_{ref}} \left[ \log \frac{π_{ref}}{π_θ} \right] ≈ \log \frac{π_{ref}}{π_θ}$
  - backward：  $\nabla_θ \text{KL}(π_{ref} | π_θ) = \mathbb{E}_{τ \sim π_{ref}} \left[ - \nabla_θ \log π_θ \right] ≈ - \nabla_θ \log π_θ$
  
- 逆向KL散度：
  - forward： $\text{KL}(π_θ | π_{ref}) = \mathbb{E}_{τ \sim π_θ} \left[ \log \frac{π_θ}{π_{ref}} \right] ≈ \log \frac{π_θ}{π_{ref}}$
  - backward： $\nabla_θ \text{KL}(π_θ | π_{ref}) = \mathbb{E}_{τ \sim π_{θ.detach}} \left[ \frac{π_{θ}}{π_{θ.detach}} \log \frac{π_θ}{π_{ref}} \nabla_θ \log π_θ \right] ≈ \frac{π_{θ}}{π_{θ.detach}} \log \frac{π_θ}{π_{ref}} \nabla_θ \log π_θ$
- 单条样本 $x$ 计算的 KL散度，即 $\log \frac{P(x)}{Q(x)}$ ，实际是“点对点信息差异”

> 注：
> - 单条样本 $\text{KL}(π_θ | π_{ref}) ≈ \log \frac{π_θ}{π_{ref}} = - \log \frac{π_{ref}}{π_θ} ≈ \text{KL}(π_{ref} | π_θ)$

## KL散度的估计

我们定义三个量：
- $k_1 = - \log \frac{π_{ref}}{π_θ}$
- $k_2 = \frac{1}{2} \left( \log \frac{π_{ref}}{π_θ} \right)^2$
- $k_3 = \frac{π_{ref}}{π_θ} - 1 - \log \frac{π_{ref}}{π_θ} $

则它们有如下性质（推导略过）
- $k_1$： 
  - forward： $\mathbb{E}_{π_{θ.detach}}[k_1] = \text{KL}(π_θ | π_{ref})$
  - backward： $\nabla_θ \mathbb{E}_{π_{θ.detach}}[k_1] = 0$
  - 性质：方差大，正向无偏，反向没有导数
- $k_2$：
  - forward： $\mathbb{E}_{π_{θ.detach}}[k_2] ≈ \text{KL}(π_θ | π_{ref})$ （有偏，但偏差很小）
  - backward： $\nabla_θ \mathbb{E}_{π_{θ.detach}}[k_2] = \nabla_θ \text{KL}(π_θ | π_{ref})$
  - 性质：方差小，正向有偏（但偏差很小，近似相等），反向无偏
- $k_3$：
  - forward： $\mathbb{E}_{π_{θ.detach}}[k_3] = \text{KL}(π_θ | π_{ref})$
  - backward： $\nabla_θ \mathbb{E}_{π_{θ.detach}}[k_3] = \nabla_θ \text{KL}(π_{ref} | π_θ) ≠ \text{KL}(π_θ | π_{ref})$
  - 性质：方差小，正向无偏，反向有偏（反向的导数不是 $\text{KL}(π_θ | π_{ref})$ ）
- $\frac{π_{θ}}{π_{θ.detach}}  k_3$：
  - forward： $\mathbb{E}_{π_{θ.detach}}[\frac{π_{θ}}{π_{θ.detach}}  k_3] = \text{KL}(π_θ | π_{ref})$
  - backward： $\nabla_θ \mathbb{E}_{π_{θ.detach}}[\frac{π_{θ}}{π_{θ.detach}}  k_3] = \nabla_θ \text{KL}(π_θ | π_{ref})$
  - 以上 $π_{θ.detach}$ 换成 $π_{old}$ 也都成了
  - 性质：**方差小，正向无偏，反向也无偏**，但是计算图比较复杂

> 备注：
> - 正向无偏：正向= $\text{KL}(π_θ | π_{ref})$
> - 反向无偏：反向= $\nabla_θ \text{KL}(π_θ | π_{ref})$

因为 $\frac{π_{θ}}{π_{θ.detach}}  k_3$ 良好的性质（正向无偏、反向也无偏，且低方差），因此GRPO中一般使用它来代替KL散度的计算
> verl中对应 `actor_rollout_ref.actor.kl_loss_type` 选项，GRPO建议使用 `k3`   
> 此处的`k3`实际指的是 $\frac{π_{θ}}{π_{θ.detach}}  k_3$ 或其离线版本 $\frac{π_{θ}}{π_{old}}  k_3$