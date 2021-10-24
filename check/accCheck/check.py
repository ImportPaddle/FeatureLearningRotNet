import os
import numpy as np
from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger
import torch,paddle

SEED=100
torch.manual_seed(SEED)
paddle.seed(SEED)
np.random.seed(SEED)
def accuracy(output, target, topk=(1,),flag='torch'):
    """Computes the precision@k for the specified values of k"""
    if flag=='torch':
        maxk = max(topk)

        batch_size = target.shape[0]
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # print(correct)
        print('------')
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)

            res.append(correct_k.mul_(100.0 / batch_size))
        return np.array([ele.data.cpu().numpy() for ele in res])
    elif flag=='paddle':
        maxk = max(topk)
        batch_size = target.shape[0]
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        target=paddle.reshape(target,(1, -1)).expand_as(pred)

        # print(pred.shape)
        # print(target.shape)
        correct = pred.equal(target)
        # print(correct)
        res = []
        for k in topk:
            correct_k = paddle.reshape(correct[:k],[-1]).numpy()
            correct_k=correct_k.sum(0)
            
            res.append((correct_k*100.0)/ batch_size)
        return np.array(res)

def torchRes(pre,target):
    pre=torch.from_numpy(pre).cuda()
    target=torch.from_numpy(target).cuda()

    res=accuracy(pre,target)
   
    return res

def paddleRes(pre,target):
    pre=paddle.to_tensor(pre)
    target=paddle.to_tensor(target)

    res=accuracy(pre,target,flag='paddle')

    return res


def main():
    reprod_log_1 = ReprodLogger()
    reprod_log_2 = ReprodLogger()

    pre=np.random.randn(128,10)
    target =np.random.randint(0,10,(128,1))

    pytorch_res=torchRes(pre,target)
    paddle_res=paddleRes(pre,target)
    
    print(type(pytorch_res))
    reprod_log_1.add("miou", pytorch_res)
    reprod_log_1.save("acc_torch.npy")

    print(type(paddle_res))
    reprod_log_2.add("miou", paddle_res)
    reprod_log_2.save("acc_paddle.npy")
def check():
    diff_helper = ReprodDiffHelper()
    info1 = diff_helper.load_info("./acc_torch.npy")
    info2 = diff_helper.load_info("./acc_paddle.npy")
    diff_helper.compare_info(info1,info2)
    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path="./diff-acc.txt")
if __name__=="__main__":
    main()
    check()