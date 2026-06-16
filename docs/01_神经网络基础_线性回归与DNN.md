# 一、神经网络基础与线性模型

> **讲练结合**：先讲清概念原理→再自测→答案折叠可遮挡学习

---

## 1.1 什么是神经网络

### 核心概念

一个神经元做的是：接收多个输入 → 加权求和 → 加偏置 → 过激活函数 → 输出。

$$ z = \sum_{i=1}^{n} w_i x_i + b $$
$$ a = f(z) $$

多个神经元组成**层**（Layer），多层堆叠即为**深度神经网络（DNN）**。

### 网络的基本结构

```
输入层（特征）→ 隐藏层1 → 隐藏层2 → ... → 输出层（结果）
```

- **输入层**：不参与计算，只是数据入口
- **隐藏层**：实际计算的地方，层数=深度
- **输出层**：产出最终结果（类别概率/回归值）

### 权重(Weight)与偏置(Bias)

- **权重 W**：控制每个输入对输出的影响程度。W 越大→该输入越重要
- **偏置 b**：平移激活函数的阈值。没有b则当所有x=0时输出必为0，模型缺乏灵活性

$$ \text{有bias: } y = Wx + b \qquad \text{无bias: } y = Wx $$

---

## 1.2 线性回归（最简单的"神经网络"）

### 原理

$$ \hat{y} = Wx + b $$

输入多个特征值 $x_1,...,x_n$，输出一个连续实数值 $\hat{y}$。

**关键特征**：无激活函数、单层、输出是实数 → **回归任务**。

### 损失函数：MSE（均方误差）

$$ \text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 $$

为什么不用交叉熵？因为交叉熵要求输入是概率分布（和为1，每个值0~1），而回归输出是任意实数。

### 代码形式

```python
class LinearRegression(nn.Module):
    def __init__(self, n_features, n_outputs=1):
        super().__init__()
        self.fc = nn.Linear(n_features, n_outputs)  # 无激活函数！

    def forward(self, x):
        return self.fc(x)

# 训练
criterion = nn.MSELoss()                # 回归用MSE
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
```

### ◆ 自测 ◆

**Q1.** 线性回归和逻辑回归的核心区别？

**Q2.** `nn.Linear(13, 1)` 中13和1分别代表什么？该层包含多少参数？

**Q3.** 回归任务为什么输出层不加激活函数？

---
<details>
<summary>答案</summary>
<p><strong>A1.</strong> 线性回归输出连续实数值（房价、温度），逻辑回归是分类（输出0~1概率）。前者用MSE损失，后者用交叉熵损失。</p>
<p><strong>A2.</strong> 13=输入特征数，1=输出值数。参数量 = 13×1 + 1(bias) = 14。</p>
<p><strong>A3.</strong> 因为回归需要输出任意实数值。如果加Sigmoid会限制输出在(0,1)，加ReLU会限制在[0,∞)，都无法正确预测可能的负值或超出范围的值。</p>
</details>

---

## 1.3 逻辑回归（二分类）

### 原理

在线性回归的基础上加一个 **Sigmoid** 激活：

$$ \hat{y} = \sigma(Wx + b) = \frac{1}{1 + e^{-(Wx+b)}} $$

输出是一个 0~1 之间的概率值，解释为"属于正类的概率"。

### 二分类交叉熵损失

$$ \text{BCE} = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})] $$

当 y=1 时只剩 $-log(\hat{y})$，预测越接近1 → loss越小；当 y=0 时只剩 $-log(1-\hat{y})$，预测越接近0 → loss越小。

### 代码形式

```python
class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.fc(x)  # 返回logits，不手动加Sigmoid

# 训练——用BCEWithLogitsLoss（内部包含Sigmoid）
criterion = nn.BCEWithLogitsLoss()
```

---

## 1.4 DNN/MLP（深度全连接网络）

### 原理

多层 Linear + 激活函数 的堆叠：

$$ h_1 = \text{ReLU}(W_1 x + b_1) $$
$$ h_2 = \text{ReLU}(W_2 h_1 + b_2) $$
$$ h_3 = \text{ReLU}(W_3 h_2 + b_3) $$
$$ \hat{y} = W_4 h_3 + b_4 $$

### 为什么需要多层？

**万能近似定理**：一个足够宽的**单隐藏层**网络理论上能拟合任何函数。但：
- "足够宽"可能指无限宽（不实际）
- 深层网络用更少的参数学到更好的表示（分层次特征：浅层学边缘→中层学形状→深层学语义）

### 参数量计算公式

$$ \text{Params}_{layer} = D_{in} \times D_{out} + D_{out} \ (\text{bias}) $$

例如 `Linear(43200, 4096)`：43200×4096 + 4096 ≈ **1.77亿参数**（仅这一层！）

### DNN处理图像的致命问题

如果把 224×224×3 的图像 Flatten 成 150528 维向量送入 DNN：
1. **参数量爆炸**：第一层 150528×hidden 就可达数亿
2. **丢失空间结构**：相邻像素的关系被破坏
3. **无平移不变性**：同一物体在图像中平移后，DNN需要重新学习

> 这就是为什么图像任务要由 CNN 来做的原因。

### ◆ 自测 ◆

**Q1.** 一个3层DNN：Linear(784→256)→ReLU→Linear(256→128)→ReLU→Linear(128→10)。总参数量？（含bias）

**Q2.** 为什么 `nn.Linear` 也叫"全连接层"（Fully Connected Layer）？

**Q3.** DNN的"深度"指的是什么？

**Q4.** DNN和MLP（多层感知机）有什么区别？

---
<details>
<summary>答案</summary>
<p><strong>A1.</strong> Layer1: 784×256+256=200,960; Layer2: 256×128+128=32,896; Layer3: 128×10+10=1,290; 总计=235,146。</p>
<p><strong>A2.</strong> 因为上一层的每个神经元都与下一层的每个神经元有连接（权重）。连接是"全"的→全连接。</p>
<p><strong>A3.</strong> "深度"指隐藏层的层数。上例中784→256→128→10有2个隐藏层（不算输入输出），深度=2。</p>
<p><strong>A4.</strong> 本质上是一回事。MLP（Multi-Layer Perceptron）是多层感知机的传统叫法，DNN（Deep Neural Network）强调"深度"足够大。在深度学习的语境下两者通常等价。</p>
</details>

---

## 1.5 回归 vs 分类

| | 回归 | 二分类 | 多分类 |
|---|------|--------|--------|
| **输出** | 一个实数 | 一个概率(0~1) | 每个类的概率(和为1) |
| **输出层激活** | 无(Linear) | Sigmoid | Softmax |
| **标签** | 连续值 | 0或1 | 0到C-1的整数 |
| **损失函数** | MSE / MAE | BCEWithLogitsLoss | CrossEntropyLoss |
| **示例** | 房价预测 | 垃圾邮件识别 | 手写数字识别 |

### ◆ 自测 ◆

**Q1.** 二分类和多分类在模型设计上的**唯一区别**是什么？

**Q2.** 如果标签是整数 `[0, 2, 1, 3]`，应该用什么损失函数？标签需要转成one-hot吗？

---
<details>
<summary>答案</summary>
<p><strong>A1.</strong> 输出层神经元数和激活函数不同。二分类：1个神经元+Sigmoid；多分类：C个神经元+Softmax。其余结构可以完全相同。</p>
<p><strong>A2.</strong> CrossEntropyLoss。不需要转one-hot，PyTorch的CrossEntropyLoss直接接受整数标签，内部自己转换。</p>
</details>

---

## 1.6 前向传播和反向传播

### 前向传播（Forward Propagation）

输入→逐层计算→输出。本质就是把数据喂进网络，按照定义好的公式算出结果。

### 反向传播（Backward Propagation / Backprop）

计算损失对每个参数的梯度，从输出层逐层往回传。

核心工具是**链式法则**：
$$ \frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial a_3} \cdot \frac{\partial a_3}{\partial z_3} \cdot \frac{\partial z_3}{\partial a_2} \cdot \frac{\partial a_2}{\partial z_2} \cdot \frac{\partial z_2}{\partial W_1} $$

### 训练循环五步法

```python
# ① 清空梯度（非常重要！否则会累积）
optimizer.zero_grad()

# ② 前向传播
output = model(x)

# ③ 计算损失
loss = criterion(output, y)

# ④ 反向传播（计算梯度）
loss.backward()

# ⑤ 更新参数
optimizer.step()
```

### ◆ 自测 ◆

**Q1.** `optimizer.zero_grad()` 如果忘记写会怎样？

**Q2.** 为什么叫"反向"传播？反的是哪个方向？

**Q3.** 前向传播和反向传播哪个耗时更多？大约比例是多少？

---
<details>
<summary>答案</summary>
<p><strong>A1.</strong> 梯度会累加。第2个batch的梯度会累加到第1个batch的梯度上，等价于batch_size翻倍。如果是有意设计就是梯度累积，如果忘了写就是bug。</p>
<p><strong>A2.</strong> 前向是数据从输入→输出（前），反向是梯度从输出损失→输入权重（后）。"反"的是计算方向。</p>
<p><strong>A3.</strong> 反向传播约是前向的2~3倍耗时（因为需要计算所有中间变量的梯度并存储）。但这是自动完成的（autograd），程序员不需要手写。</p>
</details>

---

## 1.7 Epoch、Batch、Iteration

| 概念 | 含义 | 例子（10000样本, batch=100） |
|------|------|----------------------------|
| **Batch** | 一次喂给模型的数据量 | 100个样本 |
| **Iteration** | 完成一个Batch的前向+反向=一次参数更新 | 10000/100=100次 |
| **Epoch** | 所有训练数据被完整使用一遍 | 包含100个Iteration |

```
1 Epoch = N 个 Iteration = N 次参数更新
N = ceil(总样本数 / batch_size)
```

### ◆ 自测 ◆

**Q1.** 总样本10000，batch=128，一个epoch几次参数更新？

**Q2.** batch_size从64改为128（其他不变），训练一个epoch的时间会怎么变化？

---
<details>
<summary>答案</summary>
<p><strong>A1.</strong> 10000/128 = 78.125 → ceil = 79 次参数更新。</p>
<p><strong>A2.</strong> 每次iteration处理更多数据→单次iteration变慢，但iteration次数减少→总时间不一定大幅变化。GPU充足时大batch利用率更高可能更快。</p>
</details>

---

## 本章要点速记

| 概念 | 一句话 |
|------|--------|
| 线性回归 | Linear(特征数→1)，无激活，MSE |
| 逻辑回归 | Linear(特征数→1) + Sigmoid，BCE |
| DNN | 多层Linear + ReLU，万能函数逼近 |
| 前向传播 | 输入→网络→输出 |
| 反向传播 | 损失→梯度→链式法则回传 |
| Epoch/Batch/Iter | 数据轮次/每批大小/更新次数 |
