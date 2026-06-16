/**
 * 深度学习公式计算器
 * 用于计算和展示各种深度学习相关的数学公式
 */

class NeuralNetworkCalculator {
  constructor() {
    this.weights = [];
    this.biases = [];
  }

  /**
   * 计算线性变换: z = Wx + b
   * @param {number[]} x - 输入向量
   * @param {number[][]} W - 权重矩阵
   * @param {number[]} b - 偏置向量
   * @returns {number[]} - 输出向量
   */
  linearTransform(x, W, b) {
    const output = [];
    for (let i = 0; i < W.length; i++) {
      let sum = b[i];
      for (let j = 0; j < x.length; j++) {
        sum += W[i][j] * x[j];
      }
      output.push(sum);
    }
    return output;
  }

  /**
   * Sigmoid 激活函数
   * σ(x) = 1 / (1 + e^(-x))
   * @param {number} x - 输入值
   * @returns {number} - 输出值 (0, 1)
   */
  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  /**
   * ReLU 激活函数
   * max(0, x)
   * @param {number} x - 输入值
   * @returns {number} - 输出值 [0, ∞)
   */
  relu(x) {
    return Math.max(0, x);
  }

  /**
   * Softmax 函数
   * @param {number[]} logits - 输入向量
   * @returns {number[]} - 概率分布
   */
  softmax(logits) {
    const max = Math.max(...logits);
    const exp = logits.map(x => Math.exp(x - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(x => x / sum);
  }

  /**
   * 计算交叉熵损失
   * @param {number[]} yTrue - 真实标签
   * @param {number[]} yPred - 预测概率
   * @returns {number} - 损失值
   */
  crossEntropy(yTrue, yPred) {
    let loss = 0;
    for (let i = 0; i < yTrue.length; i++) {
      const p = Math.max(yPred[i], 1e-15);
      loss -= yTrue[i] * Math.log(p);
    }
    return loss;
  }

  /**
   * 矩阵乘法
   * @param {number[][]} A - 矩阵 A
   * @param {number[][]} B - 矩阵 B
   * @returns {number[][]} - 结果矩阵
   */
  matrixMultiply(A, B) {
    const rowsA = A.length;
    const colsA = A[0].length;
    const rowsB = B.length;
    const colsB = B[0].length;
    
    if (colsA !== rowsB) {
      throw new Error('矩阵维度不匹配');
    }

    const result = [];
    for (let i = 0; i < rowsA; i++) {
      result[i] = [];
      for (let j = 0; j < colsB; j++) {
        let sum = 0;
        for (let k = 0; k < colsA; k++) {
          sum += A[i][k] * B[k][j];
        }
        result[i][j] = sum;
      }
    }
    return result;
  }

  /**
   * 计算注意力分数
   * Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
   * @param {number[][]} Q - 查询矩阵
   * @param {number[][]} K - 键矩阵
   * @param {number[][]} V - 值矩阵
   * @returns {number[][]} - 注意力输出
   */
  attention(Q, K, V) {
    const d_k = Q[0].length;
    
    // QK^T
    const K_T = K[0].map((_, colIndex) => K.map(row => row[colIndex]));
    const QK_T = this.matrixMultiply(Q, K_T);
    
    // 除以 sqrt(d_k)
    const scaled = QK_T.map(row => row.map(val => val / Math.sqrt(d_k)));
    
    // Softmax
    const weights = scaled.map(row => this.softmax(row));
    
    // 乘以 V
    return this.matrixMultiply(weights, V);
  }
}

/**
 * 主函数 - 演示各种深度学习计算
 */
function main() {
  console.log('=== 深度学习公式计算器 ===\n');
  
  const calculator = new NeuralNetworkCalculator();
  
  // 1. 测试线性变换
  console.log('1. 线性变换 z = Wx + b');
  const x = [1, 2, 3];
  const W = [[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]];
  const b = [0.1, 0.2];
  const z = calculator.linearTransform(x, W, b);
  console.log(`   输入 x: [${x}]`);
  console.log(`   线性变换结果 z: [${z.map(v => v.toFixed(4))}]\n`);
  
  // 2. 测试激活函数
  console.log('2. 激活函数测试');
  const testValue = 0.5;
  console.log(`   Sigmoid(${testValue}) = ${calculator.sigmoid(testValue).toFixed(4)}`);
  console.log(`   ReLU(${testValue}) = ${calculator.relu(testValue).toFixed(4)}`);
  console.log(`   ReLU(-${testValue}) = ${calculator.relu(-testValue).toFixed(4)}\n`);
  
  // 3. 测试 Softmax
  console.log('3. Softmax 函数');
  const logits = [2.0, 1.0, 0.1];
  const probs = calculator.softmax(logits);
  console.log(`   输入 logits: [${logits}]`);
  console.log(`   输出概率: [${probs.map(p => p.toFixed(4))}]`);
  console.log(`   概率和: ${probs.reduce((a, b) => a + b, 0).toFixed(4)}\n`);
  
  // 4. 测试交叉熵损失
  console.log('4. 交叉熵损失');
  const yTrue = [0, 1, 0];
  const yPred = [0.1, 0.8, 0.1];
  const loss = calculator.crossEntropy(yTrue, yPred);
  console.log(`   真实标签: [${yTrue}]`);
  console.log(`   预测概率: [${yPred}]`);
  console.log(`   损失值: ${loss.toFixed(4)}\n`);
  
  // 5. 测试注意力机制
  console.log('5. 注意力机制');
  const Q = [[1, 0], [0, 1], [1, 1]];
  const K = [[1, 0], [0, 1]];
  const V = [[1, 2], [3, 4]];
  const attentionOutput = calculator.attention(Q, K, V);
  console.log(`   Q 矩阵: [[1,0], [0,1], [1,1]]`);
  console.log(`   K 矩阵: [[1,0], [0,1]]`);
  console.log(`   V 矩阵: [[1,2], [3,4]]`);
  console.log(`   注意力输出:`);
  attentionOutput.forEach(row => {
    console.log(`      [${row.map(v => v.toFixed(4))}]`);
  });
  
  console.log('\n=== 计算完成 ===');
}

// 导出模块
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    NeuralNetworkCalculator,
    main
  };
}

// 如果直接运行此文件，则执行主函数
if (require.main === module) {
  main();
}
