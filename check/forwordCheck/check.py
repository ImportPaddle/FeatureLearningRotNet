import numpy as np
import os
import sys
from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger
# sys.path.append('/home/haitaowu/paddle/FeatureLearningRotNet/pytorchRotNet/architectures')
# sys.path.append('/home/haitaowu/paddle/FeatureLearningRotNet/pytorchRotNet/architectures/NonLinearClassifier.py')

# print(os.getcwd())  ##调用目录
DIR=os.path.split(os.path.realpath(__file__))[0]
# print(os.path.split(os.path.realpath(__file__))[0])
sys.path.append(os.path.join(DIR,'../../'))
# sys.path.append(os.path.join(DIR,'../../pytorchRotNet/architectures/NonLinearClassifier.py'))

# sys.path.append(os.path.join(DIR,'../../paddleRotNet/architectures'))
# sys.path.append(os.path.join(DIR,'../../paddleRotNet/architectures/NonLinearClassifier.py'))


#import torch RotNet
from pytorchRotNet.architectures import NetworkInNetwork as NetworkInNetwork
from pytorchRotNet.architectures import NonLinearClassifier

#import torch RotNet
from paddleRotNet.architectures import NetworkInNetwork as ext
from paddleRotNet.architectures import NonLinearClassifier as cla


import torchvision
# import encoding
import torch,os
import paddle


SEED=100
torch.manual_seed(SEED)
paddle.seed(SEED)
np.random.seed(SEED)

def paddleRes(net, data,out_feat_keys=None):
    net.eval()
    data = paddle.to_tensor(data)
    if out_feat_keys:
        res = net(data,out_feat_keys=out_feat_keys)
    else:
        res = net(data)
    return res.numpy()


def torchRes(net, data,out_feat_keys=None):
    net.eval()
    data = torch.from_numpy(data)
    if out_feat_keys:
        res = net(data,out_feat_keys=out_feat_keys)
    else:
        res = net(data)
    return res.data.cpu().numpy()


def get():
    opt = {'num_classes': 4, 'num_stages': 4, 'use_avg_on_conv3': False}
    model_ext_pytorch = NetworkInNetwork.create_model(opt)
    torch.save(model_ext_pytorch.state_dict(),"./model_ext_pytorch.pth")

    opt = {'num_classes': 10, 'nChannels': 192, 'cls_type': 'NIN_ConvBlock3'}
    model_cla = NonLinearClassifier.create_model(opt)  # out 128,10
    torch.save(model_cla.state_dict(), './model_cla_pytorch.pth')

def trans():
    path = './model_ext_pytorch.pth'
    torch_dict = torch.load(path)
    paddle_dict = {}
    for key in torch_dict:
        weight = torch_dict[key].cpu().detach().numpy()
        # print(key)
        if key == 'fc.weight' or key == '_feature_blocks.4.Classifier.weight':
            weight = weight.transpose()
        key = key.replace('running_mean', '_mean')
        key = key.replace('running_var', '_variance')
        paddle_dict[key] = weight
    paddle.save(paddle_dict, './model_ext_paddle.pdparams')

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
    data_ext_1 = np.random.rand(128, 3, 32, 32).astype(np.float32)
    data_ext_2 = np.random.rand(128, 3, 32, 32).astype(np.float32)
    

    ##torch ext
    optExtTorch = {'num_classes': 4, 'num_stages': 4, 'use_avg_on_conv3': False,'out_feat_keys':['conv2']}
    model_ext_pytorch = NetworkInNetwork.create_model(optExtTorch)
    state = torch.load("./model_ext_pytorch.pth")
    model_ext_pytorch.load_state_dict(state)

    ##torch cla
    optClaTorch = {'num_classes': 10, 'nChannels': 192, 'cls_type': 'NIN_ConvBlock3'}
    model_cla_pytorch = NonLinearClassifier.create_model(optClaTorch)
    state = torch.load("./model_cla_pytorch.pth")
    model_cla_pytorch.load_state_dict(state)

    ##paddle ext
    print('load paddle ext')
    optExtPaddle = {'num_classes': 4, 'num_stages': 4, 'use_avg_on_conv3': False,'out_feat_keys':['conv2']}
    model_ext_paddle = ext.create_model(optExtPaddle)
    state = paddle.load("./model_ext_paddle.pdparams")
    model_ext_paddle.set_state_dict(state)

    ##paddle cla
    print('load paddle cla')
    optClaPaddle= {'num_classes': 10, 'nChannels': 192, 'cls_type': 'NIN_ConvBlock3'}
    model_cla_paddle = cla.create_model(optClaPaddle)
    state = paddle.load("./model_cla_paddle.pdparams")
    model_cla_paddle.set_state_dict(state)

    out_feat_keys=['conv2']

    pytorch_ext_res1 = torchRes(model_ext_pytorch, data_ext_1,out_feat_keys=out_feat_keys)
    pytorch_ext_res2 = torchRes(model_ext_pytorch, data_ext_2,out_feat_keys=out_feat_keys)

    print(pytorch_ext_res1.shape)
    pytorch_cla_res1 = torchRes(model_cla_pytorch, pytorch_ext_res1)
    pytorch_cla_res2 = torchRes(model_cla_pytorch, pytorch_ext_res2)
    print(pytorch_cla_res1.shape)

    paddle_ext_res1 = paddleRes(model_ext_paddle, data_ext_1,out_feat_keys=out_feat_keys)
    paddle_ext_res2 = paddleRes(model_ext_paddle, data_ext_2,out_feat_keys=out_feat_keys)
    paddle_cla_res1 = paddleRes(model_cla_paddle, paddle_ext_res1)
    paddle_cla_res2 = paddleRes(model_cla_paddle, paddle_ext_res2)

    reprod_log_1.add("model_ext1", pytorch_ext_res1)
    reprod_log_1.add("model_ext2", pytorch_ext_res2)
    reprod_log_1.add("model_cla1", pytorch_cla_res1)
    reprod_log_1.add("model_cla2", pytorch_cla_res2)
    reprod_log_1.save("net_pytorch.npy")

    reprod_log_2.add("model_ext1", paddle_ext_res1)
    reprod_log_2.add("model_ext2", paddle_ext_res2)
    reprod_log_2.add("model_cla1", paddle_cla_res1)
    reprod_log_2.add("model_cla2", paddle_cla_res2)
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