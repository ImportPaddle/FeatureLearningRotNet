import numpy as np
from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger

def get():
    reprod_logger1 = ReprodLogger()
    reprod_logger1.add("top1", np.array([91.1600], dtype="float32"))
    reprod_logger1.save("train_align_benchmark.npy")

    reprod_logger2 = ReprodLogger()
    reprod_logger2.add("top1", np.array([91.1540], dtype="float32"))
    reprod_logger2.save("train_align_paddle.npy")
def check():
    diff_helper = ReprodDiffHelper()
    info1 = diff_helper.load_info("./train_align_benchmark.npy")
    info2 = diff_helper.load_info("./train_align_paddle.npy")
    diff_helper.compare_info(info1,info2)
    diff_helper.report(
        diff_method="mean", diff_threshold=0.015, path="./diff-train.txt")
if __name__=='__main__':
    get()
    check()

