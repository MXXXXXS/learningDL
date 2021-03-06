const network = require(`./network`)
const weights = require(`./weights`)
const n = 0.00001
module.exports = {
  layers: function() {
    const config = [
    {
      cellsNum: 784,
      activator: network.activator,
      b: 0,
      outputSize: 40,
      n: n,
      ws: weights[0]
    },
    {
      cellsNum: 40,
      activator: network.activator,
      b: 0,
      outputSize: 10,
      n: n,
      ws: weights[1]
    },
  ]
    return config.map(config => new network.layer(config))
  },
  finalLayer: function() {
    return new network.softMaxWithLoss()
  }
}