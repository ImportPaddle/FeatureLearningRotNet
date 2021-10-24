import numpy as np
import os
import sys
from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger

DIR = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(DIR, '../../'))

# import torch RotNet
from pytorchRotNet.architectures import NetworkInNetwork as NetworkInNetwork
from pytorchRotNet.architectures import NonLinearClassifier

# import torch RotNet
from paddleRotNet.architectures import NetworkInNetwork as ext
from paddleRotNet.architectures import NonLinearClassifier as cla

import torch
import paddle
SEED=100
torch.manual_seed(SEED)
paddle.seed(SEED)
np.random.seed(SEED)

def get():

    opt = {'num_classes': 10, 'nChannels': 192, 'cls_type': 'NIN_ConvBlock3'}
    model_cla = NonLinearClassifier.create_model(opt)  # out 128,10
    torch.save(model_cla.state_dict(), './model_cla_pytorch.pth')

def trans():
    path = './model_cla_pytorch.pth'
    torch_dict = torch.load(path)
    paddle_dict = {}
    for key in torch_dict:
        weight = torch_dict[key].cpu().detach().numpy()
        # print(key)
        if key == 'fc.weight' or key == 'classifier.Liniear_F.weight':
            weight = weight.transpose()
        key = key.replace('running_mean', '_mean')
        key = key.replace('running_var', '_variance')
        paddle_dict[key] = weight
    paddle.save(paddle_dict, './model_cla_paddle.pdparams')


def main():
    reprod_log_1 = ReprodLogger()
    reprod_log_2 = ReprodLogger()
    # ----------------------------------------数据----------------------------------------
    data_ext_1 = np.random.rand(128, 192, 8, 8).astype(np.float32)
    data_ext_2 = np.random.rand(128, 192, 8, 8).astype(np.float32)

    target_cla_1 = np.random.randint(0, 10, (128, 1))
    target_cla_2 = np.random.randint(0, 10, (128, 1))

    dataloader_cla = [(data_ext_1, target_cla_1), (data_ext_2, target_cla_2)]
    # ----------------------------------------模型----------------------------------------

    
    ##torch cla
    opt = {'num_classes': 10, 'nChannels': 192, 'cls_type': 'NIN_ConvBlock3'}
    model_cla_pytorch = NonLinearClassifier.create_model(opt)
    state = torch.load("./model_cla_pytorch.pth")
    model_cla_pytorch.load_state_dict(state)

    ##paddle cla
    print('load paddle cla')
    opt = {'num_classes': 10, 'nChannels': 192, 'cls_type': 'NIN_ConvBlock3'}
    model_cla_paddle = cla.create_model(opt)
    state = paddle.load("./model_cla_paddle.pdparams")
    model_cla_paddle.set_state_dict(state)
    # ----------------------------------------损失函数----------------------------------------

    criterion_torch = torch.nn.CrossEntropyLoss()
    criterion_paddle = paddle.nn.CrossEntropyLoss()
    # ----------------------------------------优化器----------------------------------------

    optim_params = {'optim_type': 'sgd', 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4, 'nesterov': True,
                    'LUT_lr': [(35, 0.1), (70, 0.02), (85, 0.004), (100, 0.0008)]}

    optim_cla_torch = torch.optim.SGD(params=model_cla_pytorch.parameters(), lr=optim_params['lr'],
                                      momentum=optim_params['momentum'],
                                      nesterov=optim_params['nesterov'] if ('nesterov' in optim_params) else False,
                                      weight_decay=optim_params['weight_decay'])
    optim_cla_paddle = paddle.optimizer.Momentum(parameters=model_ext_paddle.parameters(),
                                                 learning_rate=optim_params['lr'],
                                                 momentum=optim_params['momentum'],
                                                 use_nesterov=optim_params['nesterov'] if (
                                                         'nesterov' in optim_params) else False,
                                                 weight_decay=optim_params['weight_decay'])
    loss_paddle = []
    loss_torch = []
    for ele in dataloader_cla:
        x, y = ele
        # paddle
        paddle_x = paddle.to_tensor(x)
        paddle_y = paddle.to_tensor(y)

    
        predicted = model_cla_paddle(paddle_x)
        # predicted = paddle.transpose(predicted, perm=[0, 2, 3, 1])
        paddle_loss = criterion_paddle(predicted, paddle_y)

        optim_cla_paddle.clear_grad()
        paddle_loss.backward()
        optim_cla_paddle.step()
        
        loss_paddle.append(paddle_loss.numpy())

        # torch
        torch_x = torch.from_numpy(x)
        torch_y = torch.from_numpy(y).view(-1)
        predicted = model_cla_pytorch(torch_x)
        # predicted = torch.transpose(predicted, perm=[0, 2, 3, 1])
        torch_loss = criterion_torch(predicted, torch_y)

        optim_cla_torch.zero_grad()
        torch_loss.backward()
        optim_cla_torch.step()
        
        loss_torch.append(torch_loss.detach().numpy())
    print(loss_torch)
    print(loss_paddle)
    # torch log
    reprod_log_1.add("model_loss1", loss_torch[0])
    reprod_log_1.add("model_loss2", loss_torch[1])
    reprod_log_1.save("net_pytorch.npy")
    # paddle log
    reprod_log_2.add("model_loss1", loss_paddle[0])
    reprod_log_2.add("model_loss2", loss_paddle[1])
    reprod_log_2.save("net_paddle.npy")


def check():
    diff_helper = ReprodDiffHelper()
    info1 = diff_helper.load_info("./net_pytorch.npy")
    info2 = diff_helper.load_info("./net_paddle.npy")

    diff_helper.compare_info(info1, info2)
    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path="./diff-model.txt")


if __name__ == "__main__":
    get()
    trans()
    main()
    check()
