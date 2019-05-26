const path = require(`path`)

module.exports = {
  mode: `production`,
  entry: path.resolve(__dirname, 'test.js'),
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'test.js'
  }
}