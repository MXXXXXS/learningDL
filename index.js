const math = require(`math.js`)
const fs = require(`fs`)
const mnist = require(`mnist`)
const dataSet = mnist.set(4000, 3000)

//正确率: 85.89529843281%
const trainingTime = 600

// const n = 0.0000000005
const n = 0.000000001
// const b = 
// {
//   input: [0,0,0,1,1, ... ,0,0], // a 784-length array of floats representing each pixel of the 28 x 28 image, normalized between 0 and 1
//   output: [0,0,0,0,0,0,1,0,0,0] // a 10-length binary array that tells which digits (from 0 to 9) is in that image
// }
const trainingSet = dataSet.training
const testSet = dataSet.test

const loss = (y, t) => 0.5 * (y - t) ** 2

const dLoss = (y, t) => y - t

class ReLU {
  constructor(x) {
    this.x = x
  }

  forward() {
    let x = this.x
    this.masked = x <= 0
    return x > 0 ? x : 0
  }

  backward() {
    return this.masked ? 0 : 1
  }
}

class Layer {
  //opt格式
  // {
  //   input: [], //二维数组, 有M个神经元, 每个神经元含N个输出, 所以本层含N个神经元
  //   activator: f, //激活函数
  //   b: Number, //偏置
  //   outputSize: Number, //对应下一层神经元数量
  //   n: Number //学习率
  // }
  constructor(opt) {
    if (opt.input.length !== 0 && opt.input[0].length !== 0) {
      this.cells = []
      //创建神经元, 前一层M个神经元每个输出N个值, N依据opt.input[0].length获得
      for (let i = 0; i < opt.input[0].length; i++) {
        const inputBuf = []
        for (let j = 0; j < opt.input.length; j++) {
          inputBuf.push(opt.input[j][i])
        }
        this.cells.push(new Cell({
          input: inputBuf,
          activator: opt.activator,
          b: opt.b,
          outputSize: opt.outputSize,
          n: opt.n
        }))
      }
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

  update(dOut) {
    this.cells.forEach((cell) => {
      cell.update(dOut)
    })
  }

  //输出为一维数组, 含N个元素, 对应本层每个神经元的反向传播的偏导
  backward(dOut) {
    return this.cells.map(cell => cell.backward(dOut))
  }
}

class Cell {
  //opt格式
  // {
  //   input: [], //一维的输入
  //   activator: f, //激活函数
  //   b: Number, //偏置
  //   outputSize: Number, //对应下一层神经元数量
  //   n: Number //学习率
  // }
  constructor(opt) {
    //input输入为一维数组, 由上一层各个神经元对本神经元的输入构成
    //将输入加和, 加上并偏置
    this.inputSum = opt.input.reduce((acc, cur) => acc += cur + opt.b, 0)
    this.activator = opt.activator
    this.b = opt.b
    this.outputSize = opt.outputSize
    this.n = opt.n
    //对下一层各个神经元的权重
    this.weights = []
    //初始化权重, 取 0~1 之间的随机数
    for (let i = 0; i < this.outputSize; i++) {
      this.weights.push(Math.random())
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

  //dOut: 上一层神经元每个的偏导构成的一个一维数组, 对应每个权重
  update(dOut) {
    this.weights = this.weights.map((w, i) => w - this.n * dOut[i])
  }
  //本神经元反向传播时的偏导
  backward(dOut) {
    return dOut.reduce((acc, cur) => acc += cur, 0) * this.activated.backward()
  }
}

class Out {
  constructor(input, target, loss, dLoss) {
    this.output = []
    this.lossFn = loss
    this.dLossFn = dLoss
    this.e = []
    this.dLoss = []
    //input和target为相同维数的一维数组
    for (let i = 0; i < input[0].length; i++) {
      const inputBuf = []
      for (let j = 0; j < input.length; j++) {
        inputBuf.push(input[j][i])
      }
      this.output.push(inputBuf.reduce((acc, cur) => acc += cur, 0))
    }
    this.output.forEach((y, i) => {
      this.e.push(loss(y, target[i]))
      this.dLoss.push(dLoss(y, target[i]))
    });
  }

  input(input, target) {
    const outputBuf = []
    const bufE = []
    const bufDLoss = []
    for (let i = 0; i < input[0].length; i++) {
      const inputBuf = []
      for (let j = 0; j < input.length; j++) {
        inputBuf.push(input[j][i])
      }
      outputBuf.push(inputBuf.reduce((acc, cur) => acc += cur, 0))
    }
    outputBuf.forEach((y, i) => {
      bufE.push(this.lossFn(y, target[i]))
      bufDLoss.push(this.dLossFn(y, target[i]))
    });
    this.e = bufE
    this.dLoss = bufDLoss
  }

  loss() {
    return this.e.reduce((acc, cur) => acc += cur, 0)
  }

  backward() {
    return this.dLoss
  }
}


const losses = []

//输入数据包装
let input = trainingSet[0].input
let target = trainingSet[0].output
input = [input]
//第零层, 输入层
const layer0 = new Layer({
  input: input,
  activator: ReLU,
  b: 0,
  outputSize: 10,
  n: n
})
const output0 = layer0.forward()
//第一层, 隐藏层
const layer1 = new Layer({
  input: output0,
  activator: ReLU,
  b: 0,
  outputSize: 10,
  n: n
})
const output1 = layer1.forward()
//第二层, 隐藏层
const layer2 = new Layer({
  input: output1,
  activator: ReLU,
  b: 0,
  outputSize: 10,
  n: n
})
const output2 = layer2.forward()
const finalOutput = new Out(output2, target, loss, dLoss)


losses.push(finalOutput.loss())

for (let i = 1; i < trainingSet.length; i++) {
  let input = trainingSet[i].input
  let target = trainingSet[i].output
  input = [input]
  for (let j = 0; j < trainingTime; j++) {
    layer0.input(input)
    const output0 = layer0.forward()
    layer1.input(output0)
    const output1 = layer1.forward()
    layer2.input(output1)
    const output2 = layer2.forward()
    finalOutput.input(output2, target)
    //记录损失
    if (losses.length > 3000)
      losses.shift()
    losses.push(finalOutput.loss())
    //开始反向传播并调整权重
    const backward0 = finalOutput.backward()
    layer2.update(backward0)
    const backward1 = layer2.backward(backward0)
    layer1.update(backward1)
    const backward2 = layer1.backward(backward1)
    layer0.update(backward2)
  }
}

fs.writeFile(`result.txt`, JSON.stringify(losses), err => {
  if (err) console.error(err)
})

//Todo: test the network
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