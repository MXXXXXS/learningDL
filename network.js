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
  //   cellsNum: Number, //神经元数量
  //   activator: f, //激活函数
  //   b: Number, //偏置
  //   outputSize: Number, //对应下一层神经元数量
  //   n: Number //学习率
  //   ws: Array  //初始化权重, 可选
  // }
  constructor(opt) {
    this.cells = []
    for (let i = 0; i < opt.cellsNum; i++) {
      if (opt.ws && opt.ws.length !== 0) {
        this.cells.push(new Cell({
          activator: opt.activator,
          b: opt.b,
          outputSize: opt.outputSize,
          n: opt.n,
          w: opt.ws[i]
        }))
      } else {
        this.cells.push(new Cell({
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

  forward() {
    return this.cells.map(cell => cell.forward())
  }

  update(d) {
    this.cells.forEach((cell) => {
      cell.update(d)
    })
  }

  backward(d) {
    return this.cells.map(cell => cell.backward(d))
  }

  outputWs() {
    return this.cells.map(cell => cell.outputW())
  }
}

class Cell {
  //opt格式
  // {
  //   activator: f, //激活函数
  //   b: Number, //偏置
  //   outputSize: Number, //对应下一层神经元数量
  //   n: Number //学习率
  //   w: Array  //初始化权重, 可选
  // }
  constructor(opt) {
    this.activator = opt.activator
    this.b = opt.b
    this.outputSize = opt.outputSize
    this.n = opt.n
    if (opt.w && opt.w.length !== 0) {
      this.w = opt.w
    } else {
      this.w = []
      for (let i = 0; i < this.outputSize; i++) {
        this.w.push(Math.random())
      }
    }
  }

  input(input) {
    this.inputSum = input.reduce((acc, cur) => acc += cur, 0)
    this.activated = new this.activator(this.inputSum)
    this.output = this.activated.forward()
  }

  forward() {
    return this.w.map(w => this.output * w + this.b)
  }

  update(d) {
    this.w = this.w.map((w, i) => w - this.n * this.output * d[i])
  }

  backward(d) {
    return d.reduce((pre, di, i) => pre += di * this.w[i], 0) * this.activated.backward() * this.inputSum
  }

  outputW() {
    return this.w
  }
}

class SoftMaxWithLoss {
  constructor() {

  }

  input(input, target) {
    const inputT = []
    for (let i = 0; i < input[0].length; i++) {
      const buf = []
      for (let j = 0; j < input.length; j++) {
        buf.push(input[j][i])
      }
      inputT.push(buf)
    }

    let a = inputT.map(val => val.reduce((pre, cur) => pre += cur, 0))
    const max = Math.max(...a)
    a = a.map(val => val - max)
    const sumA = a.reduce((pre, cur) => pre += Math.exp(cur), 0)
    this.y = a.map(a => Math.exp(a) / sumA)
    this.t = target
  }

  ys() {
    return this.y
  }

  output() {
    return this.y.indexOf(Math.max(...this.y))
  }

  backward() {
    return this.y.map((y, i) => y - this.t[i])
  }

  sumLoss() {
    return this.y.reduce((pre, y, i) => pre += -1 * this.t[i] * Math.log(y), 0)
  }
}

module.exports = {
  layer: Layer,
  activator: ReLU,
  softMaxWithLoss: SoftMaxWithLoss
}