const mnist = require(`mnist`)
const dataSet = mnist.set(0, 10)
const testSet = dataSet.test

let networkConfig
try {
  networkConfig = require(`./networkConfig.js`)
} catch (error) {
  console.log(`没有配置网络载入`)
  process.exit()
}

let right = 0
let wrong = 0
const layers = networkConfig.layers()
const finalLayer = networkConfig.finalLayer()
for (let i = 0; i < testSet.length; i++) {
  let input = testSet[i].input
  let target = testSet[i].output
  // input = [input.map(val => val > 0 ? 1 : 0)]
  input = [input]
  
  const layersOutput = layers.reduce((preLayer, curLayer) => {
    curLayer.input(preLayer)
    return curLayer.forward()
  }, [input])
  finalLayer.input(layersOutput, target)
  console.log(finalLayer.ys() + `: ` + target)
  target[finalLayer.output()] === 1 ? right++ : wrong++
}
console.log(`正确率: ${right / (right + wrong) * 100}%`)