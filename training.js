const fs = require(`fs`)
const mnist = require(`mnist`)
const Progress = require(`progress`)

const trainingTime = 400
const trainingSetAmount = 6000
const dataSet = mnist.set(trainingSetAmount, 0)
const trainingSet = dataSet.training
const bar = new Progress(`Progress: :current/:total | Estimated completion time: :eta s`, { total: trainingTime })

let networkConfig
try {
  networkConfig = require(`./networkConfig.js`)
} catch (error) {
  console.error(error)
  console.log(`没有配置网络载入`)
  process.exit()
}

const layers = networkConfig.layers()
const finalLayer = networkConfig.finalLayer()

const sumLoss = []

for (let j = 0; j < trainingTime; j++) {
  for (let i = 0; i < trainingSet.length; i++) {
    let input = trainingSet[i].input
    let target = trainingSet[i].output
    const layersOutput = layers.reduce((preLayer, curLayer) => {
      curLayer.input(preLayer)
      return curLayer.forward()
    }, [input])

    finalLayer.input(layersOutput, target)
    const backward0 = finalLayer.backward()

    if (sumLoss.length > 3000)
      sumLoss.shift()
    sumLoss.push(finalLayer.sumLoss())

    layers.reverse().reduce((preD, curLayer) => {
      curLayer.update(preD)
      return curLayer.backward(preD)
    }, backward0)
    layers.reverse()
  }
  bar.tick()
}

//收集并保存整个网络的权重
const saveWeights = layers.map(layer => layer.outputWs())
fs.writeFile(`weights.js`, `module.exports = ${JSON.stringify(saveWeights)}`, err => {
  if (err) console.error(err)
})

//收集并保存每一轮的总损失
fs.writeFile(`sumLoss.json`, `${JSON.stringify(sumLoss)}`, err => {
  if (err) console.error(err)
})