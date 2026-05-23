# 五、Transformer 与多模态融合

> **讲练结合**：Self-Attention→Multi-Head→Encoder/Decoder→自回归→多模态

---

## 5.1 为什么需要Transformer？

RNN的痛点：
1. **串行计算**：第t步必须等第t-1步算完 → 无法并行 → 训练慢
2. **长距离依赖**：即使LSTM，超过100步的关系也很难学好

Transformer用**Self-Attention**一步到位，任意两位置直接计算相关性 → 并行 + 长距离依赖都解决。

---

## 5.2 Self-Attention（自注意力）

### 公式

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V $$

### 通俗理解

每个词发出三个信号：
- **Q（Query，查询）**："我想找什么样的信息？"
- **K（Key，键）**："我身上有什么信息标签？"
- **V（Value，值）**："我的实际内容是什么？"

计算过程：Q 和所有 K 做内积（相似度）→ Softmax 归一化 → 用这些权重去加权聚合 V。

### 为什么要除以 √d_k？

Q·K^T 的方差 ≈ d_k（点积是 d_k 个独立随机变量之和）。除以 √d_k 将方差稳定在 1，避免进入 Softmax 的饱和区（梯度趋近0）。

### Self vs Cross Attention

| | Q来源 | K, V来源 | 用途 |
|---|-------|----------|------|
| **Self-Attention** | 自己 | 自己 | Encoder内部、Decoder自回归 |
| **Cross-Attention** | Decoder | Encoder | 翻译：解码器关注源语言 |
| **Masked Self-Attn** | 自己 | 自己+Causal Mask | 自回归生成（不能看未来） |

### ◆ 自测 ◆

**Q1.** 输入序列长度L=10, d_k=64 → QK^T 的形状是什么？Softmax 之后的形状？

**Q2.** 为什么说Self-Attention是"置换不变"的？这为什么是个问题？

**Q3.** Self-Attention的计算复杂度？为什么长序列上很慢？

---
<details><summary>答案</summary>

**A1.** Q=(B,10,64), K^T=(B,64,10) → scores=(B,10,10)。Softmax在dim=-1上做→形状不变(10,10)，其中scores[i][j]=位置i对位置j的关注权重。

**A2.** 置换不变：如果把输入序列打乱（如"我 爱 你"→"你 爱 我"），Self-Attention每步只关注其他token的内容而非位置，输出也会跟着打乱。问题：模型无法区分"我爱你"和"你爱我"。→需要位置编码解决。

**A3.** O(L²·d)。QK^T 产生 L×L 的注意力矩阵，长序列（如L=10000）时 L²=1亿，计算和显存都吃不消。→需要稀疏注意力、FlashAttention等优化。
</details>

---

## 5.3 Multi-Head Attention（多头注意力）

### 架构

```
输入 (B, L, 512)
    ↓ 拆成 h=8 个头
Head1: (B, L, 64) → 独立 Q,K,V → Attention → (B, L, 64)
Head2: (B, L, 64) → 独立 Q,K,V → Attention → (B, L, 64)
...
Head8: (B, L, 64) → 独立 Q,K,V → Attention → (B, L, 64)
    ↓ 拼接所有头
    ↓ Linear(512→512) 输出投影
输出: (B, L, 512)
```

### 为什么叫"多头"？

不同头关注不同的模式：
- 有的头关注**句法关系**（如形容词和它修饰的名词）
- 有的头关注**语义关系**（如代词和它指代的对象）
- 有的头关注**位置关系**（如相邻词）

### 关键设计

$$ d_k = d_{model} / h $$

例如 d_model=512, h=8 → d_k=64。保证拼接后的总维度 = 8×64 = 512 = d_model，可以堆叠多层（残差连接要求维度一致）。

### ◆ 自测 ◆

**Q1.** d_model=256, num_heads=4 → 每个头的d_k=？

**Q2.** 如果不做Multi-Head（即h=1），就是普通的Self-Attention吗？

**Q3.** 为什么需要输出投影层 `W_o`？

---
<details><summary>答案</summary>

**A1.** 256/4 = 64。

**A2.** 是的。单头就是普通Self-Attention。多头通过多个不同投影子空间让模型同时关注多种不同的模式，效果显著优于单头。

**A3.** W_o 将拼接后各头独立计算的结果进行跨头信息交互和重新组合。没有 W_o，各头的输出只是简单拼在一起，缺乏综合。数学上 W_o 是必需的。
</details>

---

## 5.4 Transformer完整结构

### Encoder Block（2个子层）

```
输入 ─→ [Multi-Head Self-Attention] ─→ Add & Norm
    │                                    │
    └── 残差 ─────────────────────────────┘
    │                                    │
    └──→ [Feed-Forward Network] ────→ Add & Norm ─→输出
              ↑ 通常 d→4d→d
```

### Decoder Block（3个子层）

```
输入 ─→ [Masked MHA (Causal)] ─→ Add & Norm
    └──→ [Cross MHA (Q=Dec, KV=Enc)] ─→ Add & Norm
    └──→ [FFN] ─→ Add & Norm ─→输出
```

### Pre-LN vs Post-LN

| | Post-LN | Pre-LN |
|---|---------|--------|
| 公式 | `LN(x + Sublayer(x))` | `x + Sublayer(LN(x))` |
| 梯度 | 需通过 LN（可能衰减） | 残差直通（保持梯度） |
| 训练 | 需要Warmup | 无需Warmup也稳定 |
| 现代模型 | 较少 | **主流**（GPT, ViT等） |

### 位置编码

Self-Attention 是位置无关的 → 需要显示注入位置信息：

```python
# 正弦编码（可外推到任意长度）
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

# 可学习编码（BERT/ViT使用）
self.pos_emb = nn.Embedding(max_len, d_model)
x = x + self.pos_emb(positions)
```

### ◆ 自测 ◆

**Q1.** Encoder 和 Decoder 各包含几个子层？

**Q2.** Pre-LN 为什么比 Post-LN 训练更稳定？（从梯度角度）

**Q3.** Transformer 的 FFN 为什么中间维度通常设为 d_model 的 4 倍？

---
<details><summary>答案</summary>

**A1.** Encoder: 2个（Self-Attn + FFN）。Decoder: 3个（Masked Self-Attn + Cross-Attn + FFN）。

**A2.** Pre-LN中残差连接是 `x + Sublayer(LN(x))`，梯度可以不经过LN直接通过残差传播（即+1通道）。Post-LN中 `LN(x + Sublayer(x))`，梯度必须经过LN归一化层，在深层会逐渐衰减。

**A3.** FFN是每个位置独立的特征变换。中间扩展到4倍增加非线性容量（在宽维度上做更多的特征组合），再压缩回去。若中间维度太小（如=d_model），非线性表达能力不足；太大则计算量大，4倍是实验得出的最佳平衡。
</details>

---

## 5.5 自回归生成

### 核心定义

$$ P(x_1,...,x_T) = \prod_{t=1}^{T} P(x_t | x_1,...,x_{t-1}) $$

每个新 token 只能以**已生成的前缀**为条件——不能偷看未来。

### Causal Mask（因果掩码）

```python
# 生成下三角矩阵（含对角线）
mask = torch.tril(torch.ones(L, L))  # 可见=1, 不可见=0
```

将上三角位置（j>i）的注意力分数设为 -∞，Softmax 后这些位置权重=0。

### 解码策略

| 策略 | 做法 | 适用 |
|------|------|------|
| **Greedy** | 每步取概率最大 | 快速但单调 |
| **Beam Search (K=5)** | 保留top-5条路径 | 翻译等需精确输出 |
| **Top-k 采样** | 从top-k个中随机选 | 增加多样性 |
| **Top-p (Nucleus)** | 累积概率超p的最小集合中随机选 | 自适应，效果好 |
| **Temperature τ** | τ>1→分布平滑（多样），τ<1→分布尖锐（确定） | 控制随机性 |

### ◆ 自测 ◆

**Q1.** GPT的训练目标和BERT的训练目标有什么区别？

**Q2.** Causal Mask 为什么是下三角矩阵？

**Q3.** 自回归生成时，第 t 步需要重新计算前 t-1 步的 Q、K、V 吗？如何优化？

---
<details><summary>答案</summary>

**A1.** GPT：Next Token Prediction（自回归，P(x_t|x_{<t})，单向）。BERT：Masked Language Model（完形填空，根据上下文预测被[MASK]的词，双向）。

**A2.** 对角线上位置i=j（可以看自己），上三角j>i（"未来"不能看），下三角j<i（"过去"可以看）。Triangular Lower matrix 正好满足这个约束。

**A3.** 不需要！优化方案 = KV-Cache：缓存前t-1步的K和V，第t步只需计算新token的QKV，对历史K和V做拼接即可。将O(t²)降为O(t)。
</details>

---

## 5.6 编码器-解码器 vs 编码器-only vs 解码器-only

| 架构 | 代表模型 | 适合任务 |
|------|----------|----------|
| **Encoder-Only** (BERT) | 双向Self-Attn | 文本分类、NER、情感分析（理解任务） |
| **Decoder-Only** (GPT) | 单向Causal Self-Attn | 文本生成、对话（生成任务） |
| **Encoder-Decoder** (T5, 原始Transformer) | 双向Enc + Cross-Attn Dec | 翻译、摘要（输入输出不同的任务） |

### ◆ 自测 ◆

**Q1.** 情感分类选什么架构？为什么？

**Q2.** GPT能做分类任务吗？怎么做？

---
<details><summary>答案</summary>

**A1.** Encoder-Only（如BERT）。分类任务只需理解输入文本，不需要生成新文本。BERT的双向注意力能看到完整上下文，比单向GPT分类效果更好。

**A2.** 能。In-Context Learning：prompt "这段话的情感是：___\n文本：xxx"，GPT自回归生成"___"处的token。效果通常比专门微调的BERT差，但胜在通用。
</details>

---

## 5.7 多模态融合（MERNet案例教学）

### 核心问题

不同模态（文本、音频、图像）在特征空间、语义粒度、重要性上不同，如何有效融合？

### 常见融合策略

| 策略 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| **简单拼接** | 把各模态特征拼成一个长向量 | 简单 | 无交互 |
| **加权求和** | 每个模态乘以可学习权重后相加 | 简单+可调节 | 仍是线性 |
| **Cross-Attention** | 一模态的Q查询另一模态的KV | **有交互**，选择性关注 | 计算量大 |
| **门控融合** | g·fused + (1-g)·text（可学习的门） | 动态选择信任哪种模态 | 门可能退化 |

### Cross-Attention 融合（核心）

```
文本特征 (B, L_t, 256)         音频特征 (B, L_a, 256)
     │                               │
     ├──Q=text, K=audio, V=audio ──→ text_from_audio
     │                               │
     └──Q=audio, K=text, V=text ──→ audio_from_text
                                     │
                        Concat → Linear → 融合特征
```

双向Cross-Attention让文本"看"音频，音频"看"文本，双向信息交互。

### ELACL（情感标签锚定对比学习）

核心思想：预训练语言模型（RoBERTa）编码每个情感词汇（"angry", "happy", "sad", "neutral"）→得到4个**固定的锚点向量**。训练时，每个样本的特征被"拉向"它对应情感标签的锚点。

$$ L_{scl} = \text{CrossEntropy}(z \cdot a^T / \tau, \ label) $$

其中 z=样本特征，a=锚点矩阵(4×768)，τ=温度(0.1)。本质是**有监督对比学习**，锚点是固定的语义参考点。

### ◆ 自测 ◆

**Q1.** 简单拼接和Cross-Attention融合的本质区别？

**Q2.** 为什么多模态模型中文本通常比音频/图像准确性高？

**Q3.** ELACL中的"锚点"（Anchor）为什么要用预训练模型生成而不是随机初始化？

---
<details><summary>答案</summary>

**A1.** 拼接：静态堆叠，模态间无信息交互。Cross-Attention：动态查询，一模态根据自身需要选择性地聚合另一模态的信息。Cross-Attention捕获了模态间的"语义对应关系"。

**A2.** 文本是离散符号系统，语义明确（词汇+语法精确定义了含义）；音频(语调/音量)和图像(像素)是连续信号，语义模糊、噪声多。在情感识别中，"我很生气"远比从语调中推断生气更可靠。

**A3.** 预训练模型的锚点有语义先验（"happy"和"sad"在特征空间自然就是分离的），给对比学习提供了好的"目标位置"。随机初始化则无任何语义信息，需要从头学习锚点之间的区分性，收敛更慢。
</details>

---

## 本章要点速记

| 概念 | 一句话 |
|------|--------|
| Self-Attention | softmax(QK^T/√d_k)·V |
| Multi-Head | 将d_model拆分h份→独立算注意力→拼接 |
| Pre-LN | x+Sublayer(LN(x))，梯度直通→训练稳定 |
| Cross-Attention | Q=Decoder, KV=Encoder |
| Causal Mask | 下三角=防止看到未来→实现自回归 |
| 位置编码 | 给位置无关的Attention注入顺序信息 |
| KV-Cache | 缓存历史KV→避免重复计算 |
| Encoder-Only | BERT，双向→文本理解 |
| Decoder-Only | GPT，自回归→文本生成 |
| Encoder-Decoder | T5，Cross-Attn→翻译/摘要 |
