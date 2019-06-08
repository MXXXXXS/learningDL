# learningDL

demo性质的mnist数据集手写数字识别的神经网络, 不借助外部库全部手动搭建.

基本属于一个读书笔记, 看的这本书: ![Alt: 深度学习入门](https://img3.doubanio.com/view/subject/l/public/s29815955.jpg)

简单的4层网络(两个隐含层)

- 网络数量: 可自定义, 见 networkConfig.js
- 激活函数: ReLU
- 求偏导方式: 反向传播
- 损失函数: SoftmaxWithLoss

***19.6.6 更新: 之前的代码都是有问题的, 真正跑通是在这次更新***

文件说明

- network.js: 网络的实现
- networkConfig.js 网络的配置
- weights.js 保存整个网络的所有权重
- traning.js 用来训练网络
- test.js 用来测试网络训练结果
- sumLoss.json 用来观察最后3000(training.js中配置)个损失函数值

使用方式

    //初始化
    npm install
    //训练, 注意: 从零开始的话, 注释掉 'ws' 项, 即不引入初始权重
    npm start
    //测试训练结果, 记得不要注释 'ws' 项, 即导入已有权重
    npm test
    //应用网络
    把 test.js 里的代码手动copy出来稍微改一下(笑)
