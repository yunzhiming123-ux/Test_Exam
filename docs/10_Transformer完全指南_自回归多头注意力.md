# Transformer 完全指南：从注意力到自回归生成

> **讲练结合**：每个知识点先讲解原理+公式+代码，随后出题自测，答案在每节末尾（学习时遮挡即可）。

---

## 一、Self-Attention（自注意力）机制

### 核心思想

Self-Attention 让序列中每个位置的词都能"看到"所有其他位置的词，根据相关性加权聚合信息。

> 通俗理解：每个词问三个问题——"我在找什么？"(Q)、"你有什么？"(K)、"你实际是什么？"(V)，然后用 Q·K 的相似度决定关注多少 V。

### 公式

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V $$

其中：
- $Q$ (Query): 查询向量 —— "我要找什么"
- $K$ (Key): 键向量 —— "我有什么标签"
- $V$ (Value): 值向量 —— "我的实际内容"
- $d_k$: Key向量的维度
- $\sqrt{d_k}$: 缩放因子，防止点积过大导致 softmax 梯度消失

### 代码实现（PyTorch）

```python
import torch
import torch.nn.functional as F

def self_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, seq_len, d_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # (B, L, L)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)  # (B, L, L)
    output = torch.matmul(attn_weights, V)    # (B, L, d_k)
    return output, attn_weights
```

### 为什么叫"Self"-Attention？

因为 Q、K、V 都来自**同一个输入序列**的不同线性投影：
```python
Q = x @ W_q   # x: 输入序列
K = x @ W_k
V = x @ W_v
```

这是与 Cross-Attention（Q 和 KV 来自不同来源）的区别。

---

### ◆ 自测题 ◆

**Q1.** Self-Attention 中 Q、K、V 分别扮演什么角色？为什么 Q 和 K 要做点积？

**Q2.** 如果不除以 $\sqrt{d_k}$（即不加缩放），当 $d_k$ 很大时会发生什么？

**Q3.** 以下代码中 `masked_fill(mask == 0, float('-inf'))` 的作用是什么？在实际应用中什么时候需要 mask？

**Q4.** Self-Attention 和 Cross-Attention 的区别是什么？各举一个使用场景。

**Q5.** Self-Attention 的计算复杂度是多少？为什么长序列上会很慢？

---
<details>
<summary><b>点击查看答案</b></summary>
<p><strong>A1.</strong> Q=查询向量(我要找什么)，K=键向量(我有什么)，V=值向量(实际内容)。Q·K^T 计算每对位置之间的相似度(注意力分数)，通过 softmax 归一化后作为权重加权聚合 V。</p>
<p><strong>A2.</strong> 当 d_k 很大时，点积 Q·K^T 的值也会很大，导致 softmax 进入饱和区（梯度接近0），模型难以学习。除以 √d_k 将方差控制为1，保持梯度稳定。</p>
<p><strong>A3.</strong> 将 mask 为0的位置的注意力分数设为 -∞，softmax 后这些位置的权重变为0（因为 e^{-∞}=0）。常用于 padding mask（忽略填充位置）和 causal mask（自回归解码时防止看到未来信息）。</p>
<p><strong>A4.</strong> Self-Attention: Q、K、V 来自同一输入（如 Transformer Encoder 中）。Cross-Attention: Q 来自解码器，K、V 来自编码器输出（如机器翻译中，解码器"关注"编码器的源语言表示）。</p>
<p><strong>A5.</strong> O(L²·d)，L 为序列长度。长序列上计算量平方级增长，这是 Transformer 的主要瓶颈。解决方案：稀疏注意力、FlashAttention、线性注意力等。</p>
</details>

---

## 二、Multi-Head Attention（多头注意力）

### 核心思想

一个注意力头只能关注一种模式，多个头并行运算可以捕获不同类型的关系（如语法关系、语义关系、位置关系等）。

### 公式（完整流程）

设 $d_{model}$ 为模型维度，$h$ 为头数：

1. **线性投影**：将输入投影到 h 组 Q, K, V
   $$ Q_i = XW^Q_i, \quad K_i = XW^K_i, \quad V_i = XW^V_i $$
   其中 $W^Q_i, W^K_i \in \mathbb{R}^{d_{model} \times d_k}$, $W^V_i \in \mathbb{R}^{d_{model} \times d_v}$，$d_k = d_v = d_{model}/h$

2. **并行计算头的注意力**：
   $$ \text{head}_i = \text{Attention}(Q_i, K_i, V_i) $$

3. **拼接 + 输出投影**：
   $$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O $$
   其中 $W^O \in \mathbb{R}^{hd_v \times d_{model}}$

### 代码实现

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads        # 每个头的维度，如 512/8 = 64
        self.n_heads = n_heads
        # 将 QKV 的投影合并为一个大矩阵，效率更高
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)  # batch_size

        # 1. 线性投影并拆分为多头: (B, L, d_model) → (B, n_heads, L, d_k)
        Q = self.W_q(q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 3. 加权求和
        out = torch.matmul(attn, V)  # (B, n_heads, L, d_k)

        # 4. 合并多头 + 输出投影: (B, n_heads, L, d_k) → (B, L, d_model)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_k * self.n_heads)
        return self.W_o(out)
```

### 维度变化详解（以 d_model=512, h=8, L=10 为例）

```
输入 x:      (B, 10, 512)
    ↓ W_q (512×512)
Q:           (B, 10, 512)
    ↓ view(B, 10, 8, 64)
    ↓ transpose(1,2)
Q:           (B, 8, 10, 64)   ← 8个头，每个头64维
    ↓ Q @ K^T
scores:      (B, 8, 10, 10)   ← 每个头的注意力矩阵
    ↓ softmax + @ V
out:         (B, 8, 10, 64)
    ↓ transpose(1,2) + view
out:         (B, 10, 512)     ← 恢复原始维度
```

### 关键设计问题

**Q: 为什么 d_k = d_model / n_heads？**

保证所有头拼接后维度 = d_model：
- 每个头输出 d_k 维
- n_heads 个头拼接 = n_heads × d_k = n_heads × (d_model/n_heads) = d_model
- 这样 MHA 的输入输出维度相同，便于堆叠多层（残差连接要求维度一致）

**Q: 为什么 Q、K 的维度必须相等（都等于 d_k）？**

因为要做点积 Q @ K^T，内积要求两个向量维度一致。

**Q: V 的维度可以不同吗？**

可以。V 的维度 d_v 可以和 d_k 不同，但通常也设为 d_model / n_heads。

---

### ◆ 自测题 ◆

**Q1.** 输入 x 的 shape 为 (32, 50, 256)，MultiHeadAttention(n_heads=8, d_model=256)：
(1) 每个头的 d_k 是多少？
(2) 经过 MHA 后输出的 shape 是多少？
(3) 注意力权重矩阵 attn 的 shape 是多少？

**Q2.** 多头注意力的"多头"体现在代码的哪个步骤？如果不使用多头（n_heads=1），代码会变成什么样？

**Q3.** 为什么需要 `self.W_o`（输出投影层）？去掉它可以吗？

**Q4.** Transformer 论文（2017）中使用 h=8，d_model=512。如果改为 h=16, d_model=512，每个头的维度会变成多少？这样做可能的好处和坏处是什么？

**Q5.** 以下说法是否正确，为什么？
> "Multi-Head Attention 就是把一个输入复制 h 份，分别算 h 次 Self-Attention，然后取平均。"

---
<details>
<summary><b>点击查看答案</b></summary>
<p><strong>A1.</strong></p>
<p>(1) d_k = 256 / 8 = 32</p>
<p>(2) (32, 50, 256) — 输入输出维度相同</p>
<p>(3) (32, 8, 50, 50) — 8个头×50查询位置×50被查位置</p>
<p><strong>A2.</strong> "多头"在 view() 和 transpose() 拆分步骤体现：将 d_model 拆成 n_heads 份。n_heads=1 时就是普通的单头自注意力，计算结果完全等价，但失去捕获多种模式的能力。</p>
<p><strong>A3.</strong> W_o 将拼接后的多头输出线性投影，给不同头的输出分配不同权重。去掉后直接拼接输出，缺少跨头信息交互，表达能力下降。不可省略。</p>
<p><strong>A4.</strong> d_k = 512/16 = 32。好处：更多头可以捕获更多种细粒度的模式；坏处：每个头的维度变小，表示能力减弱，且计算量增加。</p>
<p><strong>A5.</strong> 错误。不是"复制"——每个头有独立的 W_q, W_k, W_v 投影矩阵，所以每个头在<strong>不同的投影子空间</strong>中计算注意力。也不"取平均"——是拼接后通过 W_o 投影，不是简单平均。</p>
</details>

---

## 三、Transformer 编码器（Encoder）

### 结构

一个 Transformer Encoder Block 包含两个子层：

```
输入 x ──→ [Multi-Head Self-Attention] ──→ Add & LayerNorm
     │                                       │
     └─── 残差连接 ───────────────────────────┘
                                              │
                                              ↓
              [Feed-Forward Network] ──→ Add & LayerNorm
                        │                      │
                        └── 残差连接 ───────────┘
                                              │
                                              ↓
                                            输出
```

### Pre-LN vs Post-LN

| | Post-LN（原始论文） | Pre-LN（现代常用） |
|---|---|---|
| 公式 | `LN(x + Sublayer(x))` | `x + Sublayer(LN(x))` |
| 梯度流 | 通过LN，可能衰减 | 直通残差，保持梯度 |
| 训练稳定性 | 需warmup | 不需warmup也稳定 |
| 当前主流 | 较少 | ✅ ViT、GPT系列等 |

### FFN（前馈网络）

$$ \text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2 $$

- 中间维度通常是 d_model 的 4 倍（如 512→2048→512）
- 作用：增加非线性表达能力（Attention 本质是加权求和，线性操作）
- 现代变体：使用 GELU 替代 ReLU，SwiGLU 等

### 完整 Encoder 代码

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN 风格（更稳定）
        attn_out = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout(attn_out)           # 残差连接
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)            # 残差连接
        return x
```

---

### ◆ 自测题 ◆

**Q1.** Transformer Encoder 中每个 Block 包含哪两个子层？它们各自的作用是什么？

**Q2.** 残差连接（Residual Connection）的公式是什么？为什么 Transformer 需要它？

**Q3.** LayerNorm 的作用是什么？为什么 Transformer 用 LayerNorm 而不是 BatchNorm？

**Q4.** FFN 为什么通常设计为 d_model → 4×d_model → d_model？如果去掉 FFN 会怎样？

**Q5.** 解释 Pre-LN 和 Post-LN 的区别，并说明为什么现代 Transformer 倾向于 Pre-LN。

---
<details>
<summary><b>点击查看答案</b></summary>
<p><strong>A1.</strong> (1) Multi-Head Self-Attention：建模序列内任意位置之间的依赖关系。(2) Feed-Forward Network：对每个位置独立地非线性变换，增强表达能力。</p>
<p><strong>A2.</strong> <code>output = x + Sublayer(x)</code>。需要是因为：(1) 缓解梯度消失——梯度可直通残差传播；(2) 让深层网络至少不差于浅层（恒等映射兜底）。</p>
<p><strong>A3.</strong> LayerNorm 对每个样本的特征维度做归一化（不依赖 batch_size），适合可变长度序列。BatchNorm 对 batch 维度归一化，序列长度不一致时统计量不稳定，且小 batch 下效果差。</p>
<p><strong>A4.</strong> 4倍扩展可增加中间层的非线性表示容量。去掉 FFN 后，Transformer Block 只剩线性加权求和（Attention本质是线性），模型退化为近似线性映射，失去学习复杂模式的能力。</p>
<p><strong>A5.</strong> Pre-LN: <code>x + Sublayer(LN(x))</code>，梯度可通过残差直通；Post-LN: <code>LN(x + Sublayer(x))</code>，梯度需经过LN。Pre-LN更稳定，训练初期不需要学习率warmup。现代模型（GPT、ViT等）多用Pre-LN。</p>
</details>

---

## 四、自回归生成与 Transformer 解码器（Decoder）

### 什么是自回归（Autoregressive）？

自回归模型是指：预测下一个 token 时，只能使用**已生成**的前文，不能偷看未来信息。

> 通俗理解：写作文时，你写第 N 个字的时候只能参考前面 N-1 个字，不能偷看后面的内容。

### 数学定义

给定序列 $x = (x_1, ..., x_T)$，自回归模型建模条件概率：

$$ P(x) = \prod_{t=1}^{T} P(x_t | x_1, ..., x_{t-1}) $$

即：整个序列的概率 = 每个 token 在其前缀条件下的概率连乘。

### Causal Mask（因果掩码）

实现自回归的关键技术。在 Self-Attention 中，用一个下三角矩阵 mask 防止位置 i 看到位置 j (j > i)：

```
Causal Mask (序列长度=4):
┌               ┐
│ 1  0  0  0   │  ← 位置0只能看自己
│ 1  1  0  0   │  ← 位置1可以看0,1
│ 1  1  1  0   │  ← 位置2可以看0,1,2
│ 1  1  1  1   │  ← 位置3可以看0,1,2,3
└               ┘
```

代码实现：
```python
def create_causal_mask(seq_len):
    """生成下三角掩码（1=可见, 0=不可见）"""
    mask = torch.tril(torch.ones(seq_len, seq_len))  # 下三角为1
    return mask  # shape: (seq_len, seq_len)
```

### 解码策略

自回归模型生成时有多种策略：

| 策略 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| **Greedy（贪心）** | 每步选概率最大的 token | 简单快速 | 容易陷入局部最优，缺乏多样性 |
| **Beam Search（集束搜索）** | 每步保留 top-K 条路径 | 比贪心更好 | 计算量大，可能生成重复内容 |
| **Top-k 采样** | 从概率最高的 k 个 token 中随机采样 | 增加多样性 | k 难以选择 |
| **Top-p (Nucleus) 采样** | 从累积概率超过 p 的最小集合中随机采样 | 自适应，效果好 | — |
| **Temperature 采样** | softmax 前除以温度 τ，τ>1 更随机，τ<1 更确定 | 控制随机性 | — |

### Encoder-Decoder 架构

Transformer 完整结构（如机器翻译）：

```
Encoder:  输入序列 → [EncoderBlock × N] → 编码器输出（context）
Decoder:  目标序列 → [DecoderBlock × N] → 输出概率分布
              ↑
          Cross-Attention(Q=Decoder, K=V=Encoder输出)
```

Decoder Block 包含**三个**子层（比 Encoder 多一个 Cross-Attention）：
1. **Masked Self-Attention**（自回归，causal mask）
2. **Cross-Attention**：Q 来自 decoder，K、V 来自 encoder
3. **Feed-Forward Network**

### 完整 Decoder 代码

```python
class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)       # 自注意力（带causal mask）
        self.cross_attn = MultiHeadAttention(d_model, n_heads)      # 交叉注意力
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 1. Masked Self-Attention（自回归）
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), 
                                             self.norm1(x), tgt_mask))
        # 2. Cross-Attention（关注编码器）
        x = x + self.dropout(self.cross_attn(self.norm2(x), enc_output, enc_output, src_mask))
        # 3. FFN
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x
```

---

### ◆ 自测题 ◆

**Q1.** 什么是"自回归"（Autoregressive）？为什么 Transformer Decoder 必须是自回归的？

**Q2.** Causal Mask 的形状是什么？它在注意力矩阵中的哪些位置设为 -∞？

**Q3.** Greedy Decoding 和 Beam Search 的区别是什么？各有什么适用场景？

**Q4.** Encoder-Decoder 架构中，Cross-Attention 的 Q、K、V 分别来自哪里？

**Q5.** 如果让你用 Transformer Decoder 做一个**无条件文本生成**（类似 GPT，不需要编码器输入），模型结构和 Encoder-Decoder 有什么区别？

**Q6.** Temperature 采样中 τ=0.5 和 τ=2.0 的区别是什么？

**Q7.** 以下哪种模型是自回归的？
A. BERT  B. GPT  C. ViT  D. SimCLR

---
<details>
<summary><b>点击查看答案</b></summary>
<p><strong>A1.</strong> 自回归：生成第 t 个 token 时只能用前 t-1 个 token，建模 P(x_t|x_{<t})。Decoder 必须自回归是因为生成任务需要逐 token 产生输出，不能提前看到未来信息。</p>
<p><strong>A2.</strong> 下三角矩阵（包含对角线）。上三角位置（即第 i 个 token 看第 j 个 token，j > i）设为 -∞，softmax 后权重为 0。</p>
<p><strong>A3.</strong> Greedy：每步选概率最大的 token（单路径）。Beam Search：每步保留 top-k 条路径（k条路径并行）。Greedy 适合需要快速生成的场景；Beam Search 适合翻译等精确性要求高的任务，但可能生成重复内容。</p>
<p><strong>A4.</strong> Q 来自 Decoder 的当前表示，K 和 V 都来自 Encoder 的输出。</p>
<p><strong>A5.</strong> GPT 是 Decoder-only 架构：没有 Encoder，没有 Cross-Attention，只有 Masked Self-Attention + FFN。因此它不需要源语言输入，直接根据前缀生成后续文本。</p>
<p><strong>A6.</strong> τ=0.5：softmax分布更尖锐（概率集中在高分token），生成更确定/保守。τ=2.0：分布更平滑（token间概率差距缩小），生成更随机/多样。</p>
<p><strong>A7.</strong> B。GPT 是自回归（左→右逐token预测），BERT 是双向（Masked Language Model），ViT 是图像编码器（不生成），SimCLR 是对比学习框架。</p>
</details>

---

## 五、位置编码（Positional Encoding）

### 为什么需要位置编码？

Self-Attention 是**置换不变**的——交换序列中两个位置的输入，各位置的输出只是跟着交换（内容不变）。这意味着 Self-Attention 本身无法感知词序。"A 爱 B" 和 "B 爱 A" 的 Attention 输出无法区分。

位置编码给每个位置注入唯一的位置信息。

### 正弦位置编码（原始 Transformer）

$$ PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) $$
$$ PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right) $$

其中 $pos$ 是位置，$i$ 是维度索引。

```python
def sinusoidal_positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()  # (L, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                         -(math.log(10000.0) / d_model))       # (d_model/2,)
    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位用sin
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位用cos
    return pe  # (seq_len, d_model)
```

### 可学习位置编码

很多现代模型（如 BERT, ViT）直接使用可学习的位置 embedding：

```python
self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
x = x + self.pos_embedding[:, :x.size(1), :]  # 直接加
```

### 相对位置编码与RoPE

- **相对位置编码**：不关心绝对位置，关心两个 token 之间的相对距离
- **RoPE**（旋转位置编码）：在 Q 和 K 上施加旋转变换，使注意力分数天然包含相对位置信息。LLaMA、Qwen 等现代 LLM 使用

---

### ◆ 自测题 ◆

**Q1.** 为什么说 Self-Attention 是"置换不变"的？用一句话解释。

**Q2.** 正弦位置编码为什么用 sin 和 cos 交替？第 pos 个位置的编码和第 pos+k 个位置的编码有什么关系？

**Q3.** 可学习位置编码和正弦位置编码各有什么优缺点？

**Q4.** 如果最大训练长度是512，但测试时来了一个长度1000的序列，哪种位置编码可以处理？哪种不行？

---
<details>
<summary><b>点击查看答案</b></summary>
<p><strong>A1.</strong> Self-Attention 的输出 = softmax(QK^T)V，QK^T 的计算对序列顺序无感知——交换两个位置只是交换对应行。位置信息必须由位置编码注入。</p>
<p><strong>A2.</strong> sin/cos 交替使用使得不同频率的正弦波叠加，不同位置有不同编码模式。因为 sin(pos+k) 可表示为 sin(pos) 和 cos(pos) 的线性组合，所以编码包含相对位置信息。</p>
<p><strong>A3.</strong> 可学习：灵活（数据驱动学习最优编码），但受训练长度限制（超过 max_len 无法外推）。正弦：固定函数、可外推到任意长度，但可能不如学习的好。</p>
<p><strong>A4.</strong> 正弦编码可以直接外推到1000（因为它是确定函数，输入任意 pos 都能算）。可学习编码如果只训练了512就不能直接处理1000（没有训练过位置512+的embedding），需要插值等技巧。</p>
</details>

---

## 六、GPT 系列：Decoder-Only 自回归语言模型

### 架构特点

GPT = Transformer Decoder 去掉 Cross-Attention（因为没有 Encoder）

```
输入 tokens → Token Embedding + Position Embedding
    → [Masked Self-Attention → FFN] × N层
    → LayerNorm → Linear(vocab_size) → Softmax
    → 输出下一个 token 的概率分布
```

### 训练：Next Token Prediction

```python
# 给定序列 "我 爱 中国"
# 输入:  [我,  爱,  中国]
# 目标:  [爱, 中国, <EOS>]   —— 每个位置预测"下一个"token
# Causal Mask 确保第i位只能看到前i个token
```

### 生成（推理）

```python
def generate(model, prompt, max_len=50, temperature=1.0):
    model.eval()
    tokens = tokenize(prompt)
    for _ in range(max_len):
        logits = model(tokens)           # (1, seq_len, vocab_size)
        next_logits = logits[0, -1, :]   # 取最后一个位置的logits
        next_logits = next_logits / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)  # 从分布中采样
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        if next_token == EOS_TOKEN:
            break
    return detokenize(tokens[0])
```

### 为什么 GPT 只有 Decoder？

- 语言建模任务本身就是**给定前缀预测下一个词**——天然适合 Decoder-Only
- 不需要外部输入（Encoder），自监督学习：用整个互联网文本，输入=前缀，标签=下一个词
- 简洁、易扩展（Scaling Law），效果随规模增大稳定提升

---

### ◆ 自测题 ◆

**Q1.** GPT 的训练目标是什么？给定 "我 爱 北京 天安门"，写出训练时的输入和标签。

**Q2.** GPT 生成时为什么只取 `logits[0, -1, :]`？为什么不取前面位置的输出？

**Q3.** GPT 和 BERT 的核心区别是什么？（提示：自回归 vs 双向、训练任务、使用场景）

**Q4.** 为什么 GPT 可以堆叠很多层（如 GPT-3 96层），而传统的 RNN/LSTM 很难做到？

---
<details>
<summary><b>点击查看答案</b></summary>
<p><strong>A1.</strong> 训练目标：Next Token Prediction（下一个token预测）。输入: [我, 爱, 北京, 天安门]，标签: [爱, 北京, 天安门, <EOS>]。每个位置的输出预测下一个token。</p>
<p><strong>A2.</strong> 生成时只需要预测<strong>下一个</strong> token，而最后一个位置汇聚了所有前文信息（通过 causal self-attention）。前面位置的输出在训练时预测各自的下一个token，推理时不需要。</p>
<p><strong>A3.</strong> GPT: 自回归(左→右)、Next Token Prediction、文本生成/对话。BERT: 双向、Masked Language Model(完形填空)、文本理解/分类。GPT适合生成任务，BERT适合理解任务。</p>
<p><strong>A4.</strong> Transformer 的 Pre-LN + 残差连接使梯度可以直通，不受深度影响。而 RNN 需要 BPTT 跨时间步反向传播，梯度连乘导致梯度消失/爆炸，层数和序列长度双重限制深度。</p>
</details>

---

## 七、综合对比：Encoder-Only / Decoder-Only / Encoder-Decoder

| 架构 | 代表模型 | 注意力类型 | 典型任务 | 自回归? |
|------|----------|-----------|----------|---------|
| **Encoder-Only** | BERT, RoBERTa | 双向 Self-Attn | 分类、NER、问答 | ❌ |
| **Decoder-Only** | GPT, LLaMA | 单向(Causal) Self-Attn | 文本生成、对话 | ✅ |
| **Encoder-Decoder** | T5, BART, 原始Transformer | 双向+Cross-Attn | 翻译、摘要 | ✅(Decoder部分)|

---

### ◆ 自测题 ◆

**Q1.** 如果任务是"判断一段文本的情感是正面还是负面"，应该选哪种架构？为什么？

**Q2.** 如果任务是"把英文翻译成中文"，应该选哪种架构？为什么？

**Q3.** Decoder-Only 模型（如GPT）能完成"文本分类"任务吗？如果能，怎么做到的？

---
<details>
<summary><b>点击查看答案</b></summary>
<p><strong>A1.</strong> Encoder-Only（如BERT）。因为文本理解/分类任务不需要生成新文本，只需要对输入进行编码后做分类。双向注意力能让模型看到完整上下文。</p>
<p><strong>A2.</strong> Encoder-Decoder（如T5）。Encoder 双向编码源语言，Decoder 自回归生成目标语言，Cross-Attention 连接两者。</p>
<p><strong>A3.</strong> 能。GPT 可通过 Prompt 方式：输入 "这段话的情感是：[MASK]\n 文本：今天天气真好"，模型自回归地生成下一个 token（如"正面"）。这就是 In-Context Learning。</p>
</details>

---

## 八、Transformer 变体与前沿

| 变体 | 核心改进 | 代表模型 |
|------|----------|----------|
| **Sparse Attention** | 只计算部分位置的注意力，降低复杂度 O(L²)→O(L·logL) | Longformer, BigBird |
| **FlashAttention** | 利用 GPU 内存层级优化注意力计算，不改变数学结果 | GPT-4, LLaMA 使用的底层优化 |
| **KV-Cache** | 推理时缓存之前生成的 K、V，避免重复计算 | 几乎所有自回归LLM |
| **MoE (混合专家)** | FFN 替换为多个专家子网络，每次只激活一部分 | Mixtral, DeepSeek-V2 |
| **MQA / GQA** | 多头共享 K、V（减少显存），Q 独立（保持效果） | LLaMA2 (GQA) |
| **SwiGLU** | FFN 使用门控线性单元替代 ReLU | LLaMA, PaLM |

### KV-Cache 详解

自回归生成时，第 t 步不需要重新计算前 t-1 步的 K 和 V：

```python
# 无缓存的低效做法（每一步都重新算整个序列）：
for i in range(max_len):
    logits = model(tokens)      # tokens越来越长，每次都重新算！
    next_token = sample(logits[-1])
    tokens = cat(tokens, next_token)

# 有缓存的优化做法：
past_kv = None
for i in range(max_len):
    logits, past_kv = model(next_token, past_kv=past_kv)  # 只算新token
    next_token = sample(logits[-1])
```

---

### ◆ 自测题 ◆

**Q1.** KV-Cache 为什么能加速推理？它缓存了什么？

**Q2.** MQA（Multi-Query Attention）和 GQA（Grouped-Query Attention）的区别是什么？

**Q3.** FlashAttention 改变了注意力机制的数学公式吗？它的加速原理是什么？

---
<details>
<summary><b>点击查看答案</b></summary>
<p><strong>A1.</strong> KV-Cache 缓存已生成 token 的 Key 和 Value。第 t 步推理时，前 t-1 步的 K、V 直接复用，只需计算新 token 的 QKV，将 O(L²·t) 减少到 O(L·t)。</p>
<p><strong>A2.</strong> MQA：所有头共享同一组 K、V（只有一组），Q 独立；GQA：头分若干组，每组内共享 K、V（折中方案）。MQA 最省显存但效果可能下降，GQA 平衡效果和效率。</p>
<p><strong>A3.</strong> 没有改变数学公式。FlashAttention 通过将注意力计算分块（tiling），在 GPU 的 SRAM（高速缓存）中完成运算，避免反复读写 HBM（显存），大幅减少 I/O 开销。</p>
</details>

---

## 总结速记表

| 概念 | 一句话 |
|------|--------|
| **Self-Attention** | 序列内每个位置加权聚合所有位置的信息：softmax(QK^T/√d_k)V |
| **Multi-Head** | 将 d_model 拆分到 h 个子空间并行计算注意力后拼接 |
| **Causal Mask** | 下三角掩码，防止看到未来信息，实现自回归 |
| **Cross-Attention** | Q来自解码器，KV来自编码器；连接两种表示 |
| **Pre-LN** | 残差前归一化：x + Sublayer(LN(x))，训练更稳定 |
| **Position Encoding** | 给 Self-Attention 注入序列顺序信息 |
| **自回归** | 逐 token 生成，P(x)=ΠP(x_t|x_{<t}) |
| **KV-Cache** | 缓存历史 KV 避免重复计算，加速自回归推理 |
