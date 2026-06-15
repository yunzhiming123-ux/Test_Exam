# 四、RNN / LSTM / GRU 循环神经网络

> **讲练结合**：原理→公式推导→代码→自测

---

## 4.1 为什么需要RNN？

CNN和DNN的输入大小是固定的，且每次处理独立样本。但很多数据是**序列**：
- 文本："我 爱 中国" — 字之间有顺序依赖
- 语音：连续的音频帧
- 股票：每天的价格形成趋势
- 视频：连续的图像帧

RNN通过**循环状态**让网络拥有"记忆"，处理任意长序列时共享同一套参数。

---

## 4.2 SimpleRNN

### 核心公式

$$ h_t = \tanh(W x_t + U h_{t-1} + b) $$
$$ y_t = V h_t + c $$

### 参数解释

| 符号 | 含义 | 维度 |
|------|------|------|
| $x_t$ | t时刻的输入向量 | (batch, input_dim) |
| $h_{t-1}$ | t-1时刻的隐藏状态（"记忆"） | (batch, hidden_dim) |
| $W$ | 输入权重矩阵 | (input_dim, hidden_dim) |
| $U$ | 循环权重矩阵（状态转移） | (hidden_dim, hidden_dim) |
| $b$ | 偏置 | (hidden_dim,) |

### 关键特征

- **权重共享**：W、U、b 在所有时间步复用——无论序列多长参数量不变
- **循环**：$h_t$ 同时依赖 $x_t$ 和 $h_{t-1}$，形成了信息跨时间传递的通道

### 展开视图

```
t=1:  x₁→[h₁]→y₁
t=2:  x₂→[h₁→h₂]→y₂     ← h₂包含x₁和x₂的信息
t=3:  x₃→[h₂→h₃]→y₃     ← h₃包含x₁,x₂,x₃的信息
```

### 代码

```python
rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
# 输入: (B, T, 128) → 输出: (B, T, 256)
```

### ◆ 自测 ◆

**Q1.** RNN的"循环"具体指什么？画图或文字描述。

**Q2.** 序列长度从10变成100，RNN的参数量变化吗？

**Q3.** SimpleRNN的激活函数为什么是Tanh而不是ReLU？

---
<details>
<summary>答案</summary>
<p><strong>A1.</strong> 循环指 $h_t$ 的计算依赖 $h_{t-1}$，$h_{t-1}$ 又依赖 $h_{t-2}$ ...→形成一条贯穿时间的链条。实现上是同一个RNN cell反复调用（for循环遍历时间步）。</p>
<p><strong>A2.</strong> 不变。W、U、b在所有时间步共享，参数量只取决于 input_dim 和 hidden_dim，与序列长度无关。这是RNN能处理变长序列的根本原因。</p>
<p><strong>A3.</strong> Tanh 输出 (-1,1)，有正有负，零中心 → 状态更新方向更平衡。ReLU 只输出非负值 → 隐藏状态会无限增长（指数级变大）→ 梯度爆炸。但实践中Tanh也带来梯度消失问题，这是SRN的主要缺陷。</p>
</details>

---

## 4.3 RNN的致命问题：梯度消失/爆炸

### BPTT（Backpropagation Through Time）

RNN的反向传播需要沿时间步逐层回传：

$$ \frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \cdot \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}} $$

其中：
$$ \frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\tanh'(·)) \cdot U^T $$

### 为什么会消失？

- $\tanh'(x)$ 最大值=1，实际值通常在0到1之间
- $|U|$ 如果 $< 1$（权重矩阵特征值<1）
- T个小于1的数连乘 → 趋近于 0（指数衰减）
- **结果**：离当前时刻越远的信息，梯度越小 → 模型学不到长距离依赖

### 为什么也会爆炸？

- 如果 $|U| > 1$，连乘 → 指数增长 → 梯度爆炸

### ◆ 自测 ◆

**Q1.** 在训练RNN时 loss 突然变成 NaN，最可能是什么原因？

**Q2.** 为什么"长距离依赖"在SRN中很难学到？

---
<details>
<summary>答案</summary>
<p><strong>A1.</strong> 梯度爆炸。某个时间步的梯度指数增长，导致参数更新过大→溢出为NaN。解决：梯度裁剪。</p>
<p><strong>A2.</strong> 在BPTT中梯度需跨T个时间步传播，每一步乘一次 tanh'(≤1)和U。当T很大时（如100），连乘后梯度几乎为0→模型无法利用远距离的信息→学不到"第1个词和第100个词的关系"。</p>
</details>

---

## 4.4 LSTM：长短期记忆

### 为什么LSTM更好？

LSTM引入**细胞状态** $c_t$，其更新接近**线性**（不像SRN需要经过tanh）：

$$ c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t $$

**反向传播时**：$\frac{\partial c_t}{\partial c_{t-1}} = f_t$（元素级乘法，不是矩阵乘法！）

当遗忘门 $f_t \approx 1$ 时，梯度无损地沿时间步传播！这就是LSTM能学长距离依赖的根本原因。

### 六个公式（必考）

```
遗忘门:   f_t = σ(W_f·x_t + U_f·h_{t-1} + b_f)     → 决定扔掉哪些旧记忆
输入门:   i_t = σ(W_i·x_t + U_i·h_{t-1} + b_i)     → 决定写入哪些新信息
候选记忆: c̃_t = tanh(W_c·x_t + U_c·h_{t-1} + b_c)   → 新信息的候选值
细胞更新: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t            → 记忆更新（线性！）
输出门:   o_t = σ(W_o·x_t + U_o·h_{t-1} + b_o)     → 决定输出什么
隐状态:   h_t = o_t ⊙ tanh(c_t)                     → 当前输出
```

**共12个参数矩阵**：4×(W, U, b)，记法: {遗忘, 输入, 候选, 输出} × {W, U, b}

### 门控的妙处

| 门 | 激活函数 | 取值范围 | 语义 |
|----|---------|----------|------|
| $f_t$ (遗忘) | Sigmoid | (0,1) | 0=全忘，1=全保留 |
| $i_t$ (输入) | Sigmoid | (0,1) | 0=不写入，1=全写入 |
| $o_t$ (输出) | Sigmoid | (0,1) | 0=不输出，1=全输出 |

所有门用Sigmoid→输出(0,1)→天然代表"门"的开闭程度。

### 代码

```python
lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, 
               batch_first=True, dropout=0.3)

# 输入: x=(B, T, 128)
# 初始状态: h0=(2, B, 256), c0=(2, B, 256)  # num_layers=2
output, (hn, cn) = lstm(x, (h0, c0))
# output: (B, T, 256) — 每个时间步的输出
# hn, cn: (2, B, 256) — 最后一个时间步的隐藏/细胞状态
```

### ◆ 自测 ◆

**Q1.** LSTM三个门各用的是什么激活函数？为什么？

**Q2.** $c_t$ 和 $h_t$ 的区别和关系？

**Q3.** 如果遗忘门永远输出 $f_t$=1，输入门永远输出 $i_t$=0，LSTM退化成什么？

**Q4.** `nn.LSTM` 中 `num_layers=2` 和 `bidirectional=True` 的区别？

**Q5.** LSTM 一共有多少组参数矩阵？写出变量名。

---
<details>
<summary>答案</summary>
<p><strong>A1.</strong> 三个门都用Sigmoid（输出0~1，"门"需要这个范围表示开闭程度）。候选记忆c̃_t用Tanh（(-1,1)，包含正负信息）。</p>
<p><strong>A2.</strong> $c_t$=长期记忆（线性更新，梯度无损跨时间步传播→解决消失问题）；$h_t$=短期输出（$c_t$经输出门过滤后对外的"发言"）。$c_t$内部用，$h_t$对外用。</p>
<p><strong>A3.</strong> 退化成普通的序列累加器：$c_t = c_{t-1}$（永远记住所有过去信息），无法选择性遗忘。实际中需要学到的策略是在需要遗忘时令$f_t$→0。</p>
<p><strong>A4.</strong> num_layers=2：两层LSTM纵向堆叠→输出维度不变(hidden_size)，参数翻倍。bidirectional=True：正向+反向各一个LSTM→输出维度翻倍(2×hidden_size)。</p>
<p><strong>A5.</strong> 12个：W_f, U_f, b_f (遗忘门); W_i, U_i, b_i (输入门); W_c, U_c, b_c (候选记忆); W_o, U_o, b_o (输出门)。</p>
</details>

---

## 4.5 GRU（门控循环单元）

### LSTM简化版

- **2个门**（不是3个）：更新门 $z_t$ + 重置门 $r_t$
- **无独立细胞状态**：$h_t$ 同时扮演 LSTM 中 $c_t$ 和 $h_t$ 的角色
- **参数量 = LSTM 的 3/4**

### 公式

```
重置门: r_t = σ(W_r·x_t + U_r·h_{t-1})        → 决定遗忘多少
更新门: z_t = σ(W_z·x_t + U_z·h_{t-1})        → 决定保留多少旧/新
候选:   h̃_t = tanh(W_h·x_t + U_h·(r_t⊙h_{t-1}))  → 新信息
输出:   h_t = (1-z_t)⊙h_{t-1} + z_t⊙h̃_t         → 新旧混合
```

### 对比

| | LSTM | GRU |
|---|------|-----|
| 门控数 | 3 | 2 |
| 细胞状态c_t | 有 | 无(合并到h_t) |
| 参数矩阵 | 12 | 9 |
| 何时选 | 大数据/需要精细控制 | 小数据/追求速度 |

---

## 4.6 Teacher Forcing

### 问题

训练RNN做序列生成时，每一步用模型上一步的预测作为输入还是真实标签？

| 策略 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| **Teacher Forcing** | 用真实标签 | 训练稳定，收敛快 | 训练-推理不一致（Exposure Bias） |
| **Free Running** | 用模型自己的预测 | 和推理行为一致 | 初期预测差→误差累积→训练慢 |

实践中通常用Teacher Forcing训练，推理时切换为Free Running。

### ◆ 自测 ◆

**Q1.** 什么是 Exposure Bias？

**Q2.** 除了Teacher Forcing，还有什么策略可以缓解Exposure Bias？

---
<details>
<summary>答案</summary>
<p><strong>A1.</strong> Teacher Forcing训练时模型每一步看到的是"完美"的真实前文，推理时则只能看到自己生成的（可能有错的）前缀。这种训练和推理条件的不一致导致推理时错误累积。</p>
<p><strong>A2.</strong> Scheduled Sampling（训练时逐步增加用自己预测的概率）、Professor Forcing（用GAN使自由运行和Teacher Forcing的隐藏状态分布一致）。</p>
</details>

---

## 本章要点速记

| 概念 | 一句话 |
|------|--------|
| SRN | h_t=tanh(Wx_t+Uh_{t-1})，梯度消失→长序列差 |
| BPTT | 反向传播沿时间步展开，链式求导连乘 |
| LSTM | 3门+细胞状态c_t→梯度可无损传播(当f_t≈1时) |
| GRU | LSTM简化版：2门+无独立c→参数量少25% |
| Teacher Forcing | 训练用真实标签→稳定但带来Exposure Bias |
| 梯度裁剪 | `clip_grad_norm_(max_norm=5)` 防止梯度爆炸 |
