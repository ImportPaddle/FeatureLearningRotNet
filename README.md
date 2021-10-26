训练：

运行sh run_cifar10_based_unsupervised_experiments.sh

- [x] 模型对齐
- [x] loss对齐
- [x] 评估指标对齐
- [x] 反向对齐
- [x] 训练对齐



训练：

RotNet_NIN4blocks训练：

*CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_RotNet_NIN4blocks*



ConvClassifier训练：

CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats



训练日志与训练模型

classifier_net_epoch92 放在./experiments/CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats

model_net_epoch102 放在./experiments/CIFAR10_RotNet_NIN4blocks

model_opt_epoch102 放在./experiments/CIFAR10_RotNet_NIN4blocks



[百度网盘](https://pan.baidu.com/s/1tPqxjbO6E3gWlcOMpqa02w)

提取码：k1gf



评估：

CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats --evaluate --checkpoint=92



模型精度：

| 模型                 | CIFAR10 Top1 acc |
| -------------------- | ---------------- |
| RotNet+conv(pytorch) | 91.16            |
| paddle               | 91.6238          |





问题1: 已解决
说明：PaddlePaddle的SGD不支持动量更新、动量衰减和Nesterov动量，这里需要使用paddle.optimizer.Momentum API实现这些功能。

```python
optimizer = paddle.optimizer.Momentum(parameters=parameters,
                                                 learning_rate=learning_rate,
                                                 momentum=optim_opts['momentum'],
                                                 use_nesterov=optim_opts['nesterov'] if (
                                                         'nesterov' in optim_opts) else False,
                                                 weight_decay=optim_opts['weight_decay'])
```

