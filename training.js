const fs = require(`fs`)
const mnist = require(`mnist`)

const trainingTime = 200
const dataSet = mnist.set(1000, 1000)
const trainingSet = dataSet.training

let networkConfig
try {
  networkConfig = require(`./networkConfig.js`)
} catch (error) {
  console.error(error)
  console.log(`没有网络载入`)
  process.exit()
}

const layers = networkConfig.layers()
const finalLayer = networkConfig.finalLayer()

for (let i = 0; i < trainingSet.length; i++) {
  let input = trainingSet[i].input
  let target = trainingSet[i].output
  for (let j = 0; j < trainingTime; j++) {
    const layersOutput = layers.reduce((preLayer, curLayer) => {
       curLayer.input(preLayer)
       return curLayer.forward()
    }, [input])

    finalLayer.input(layersOutput, target)
    const backward0 = finalLayer.backward()
  
    layers.reverse().reduce((preD, curLayer) => {
      curLayer.update(preD)
      return curLayer.backward(preD)
    }, backward0)
    layers.reverse()
  }
}

//收集并保存整个网络的权重
const saveWeights = layers.map(layer => layer.outputWs())
fs.writeFile(`weights.js`, `module.exports = ${JSON.stringify(saveWeights)}`, err => {
  if (err) console.error(err)
})