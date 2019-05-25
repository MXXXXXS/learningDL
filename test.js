const fs = require(`fs`)
const mnist = require(`mnist`)
const dataSet = mnist.set(2000, 1000)
const testSet = dataSet.test

let weights = fs.readFileSync(`weights.json`)
if (weights) {
  weights = JSON.parse(weights)
} else {
  console.log(`权重输入失败`)
  process.exit()
}
const loss = (y, t) => 0.5 * (y - t) ** 2

class ReLU {
  constructor(x) {
    this.x = x
  }

  forward() {
    let x = this.x
    this.masked = x <= 0
    return x > 0 ? x : 0
  }
}

class Layer {
  //opt格式
  // {
  //   cellsNum: Number, 
  //   activator: f, //激活函数
  //   b: Number, //偏置
  //   outputSize: Number, //对应下一层神经元数量
  //   ws: Array  //初始化权重, 可选
  // }
  constructor(opt) {
    this.cells = []
    //创建神经元, 前一层M个神经元每个输出N个值, N依据opt.cellsNum获得
    for (let i = 0; i < opt.cellsNum; i++) {
      this.cells.push(new Cell({
        activator: opt.activator,
        b: opt.b,
        outputSize: opt.outputSize,
        n: opt.n,
        w: opt.ws[i]
      }))
    }
  }

  input(input) {
    for (let i = 0; i < input[0].length; i++) {
      const inputBuf = []
      for (let j = 0; j < input.length; j++) {
        inputBuf.push(input[j][i])
      }
      this.cells[i].input(inputBuf)
    }
  }

  //输出为二维数组, 含N个元素, 对应本层神经元数量
  //每个元素为一个一维数组, 对应每个神经元outputSize个输出
  forward() {
    return this.cells.map(cell => cell.forward())
  }
}

class Cell {
  //opt格式
  // {
  //   activator: f, //激活函数
  //   b: Number, //偏置
  //   outputSize: Number, //对应下一层神经元数量
  //   w: Array  //初始化权重, 可选
  // }
  constructor(opt) {
    //input输入为一维数组, 由上一层各个神经元对本神经元的输入构成
    //将输入加和, 加上并偏置
    this.activator = opt.activator
    this.b = opt.b
    this.outputSize = opt.outputSize
    this.n = opt.n
    //初始化权重
    if (opt.w) {
      this.weights = opt.w
    } else {
      //对下一层各个神经元的权重
      this.weights = []
      for (let i = 0; i < this.outputSize; i++) {
        this.weights.push(Math.random())
      }
    }
    //激活函数对输入作用后的值
    this.activated = new this.activator(this.inputSum)
    this.forwardVal = this.activated.forward()
  }

  //改变输入
  input(input) {
    this.inputSum = input.reduce((acc, cur) => acc += cur + this.b, 0)
    this.activated = new this.activator(this.inputSum)
    this.forwardVal = this.activated.forward()
  }

  //输出一维数组, 由激活后的值与各个权重相乘
  forward() {
    return this.weights.map(w => this.forwardVal * w)
  }

  inputWeights(w) {
    this.weights = w
  }
}

class Out {
  constructor(loss) {
    this.lossFn = loss
  }

  input(input, target) {
    const outputBuf = []
    const bufE = []
    for (let i = 0; i < input[0].length; i++) {
      const inputBuf = []
      for (let j = 0; j < input.length; j++) {
        inputBuf.push(input[j][i])
      }
      outputBuf.push(inputBuf.reduce((acc, cur) => acc += cur, 0))
    }
    outputBuf.forEach((y, i) => {
      bufE.push(this.lossFn(y, target[i]))
    });
    this.e = bufE
  }

  loss() {
    return this.e.reduce((acc, cur) => acc += cur, 0)
  }
}
//第零层, 输入层
const layer0 = new Layer({
  cellsNum: 784,
  activator: ReLU,
  b: 0,
  outputSize: 10,
  ws: weights[0]
})
//第一层, 隐藏层
const layer1 = new Layer({
  cellsNum: 10,
  activator: ReLU,
  b: 0,
  outputSize: 10,
  ws: weights[1]
})
//第二层, 隐藏层
const layer2 = new Layer({
  cellsNum: 10,
  activator: ReLU,
  b: 0,
  outputSize: 10,
  ws: weights[2]
})
const finalOutput = new Out(loss)

let right = 0
let wrong = 0
for (let i = 1; i < testSet.length; i++) {
  let input = testSet[i].input
  let target = testSet[i].output
  input = [input]
  layer0.input(input)
  const output0 = layer0.forward()
  layer1.input(output0)
  const output1 = layer1.forward()
  layer2.input(output1)
  const output2 = layer2.forward()
  finalOutput.input(output2, target)
  target[finalOutput.e.indexOf(Math.max(...finalOutput.e))] === 1 ? right++ : wrong++
}
console.log(`正确率: ${right / (right + wrong) * 100}%`)