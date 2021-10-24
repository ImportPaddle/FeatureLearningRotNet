1.模型对齐 yes

Torch -> Paddle 

坑：nn.AvgPool2D(exclusive=False,kernel_size=3, stride=2, padding=1）

exclusive=False

paddle默认与torch相反

运行check/forwordCheck/check.py

diff:

ext1: 5.027723304351639e-08

fla1: 1.1689111545365449e-07



2.loss对齐 yes

运行check/forwordCheck/check.py

diff: 0.0

3.评估指标对齐 yes

运行check/accCheck/check.py

diff: 0.0



4.反向对齐 yes

运行check/forwordCheck/acc.py



5.训练对齐



问题1: 



torch.optimizer.SGD(parameters=parameters,lr=learning_rate,



​                momentum=optim_opts['momentum'],



​                nesterov=optim_opts['nesterov'] if ('nesterov' in optim_opts) else False,



​                weight_decay=optim_opts['weight_decay'])



paddle无momentum和nesterov超参数



说明：PaddlePaddle的SGD不支持动量更新、动量衰减和Nesterov动量，这里需要使用paddle.optimizer.Momentum API实现这些功能。



paddle.optimizer.Momentum(parameters=parameters,lr=learning_rate,



​                momentum=optim_opts['momentum'],



​                nesterov=optim_opts['nesterov'] if ('nesterov' in optim_opts) else False,



​                weight_decay=optim_opts['weight_decay'])





问题2:

torch 中num_batches_tracked 参数未设置，导致模型未对齐