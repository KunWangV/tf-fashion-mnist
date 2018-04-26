# tf-fashion-mnist

1、文件读写 file_random.py： 使用shell命令统计文件行数、使用蓄水池算法选择行、多线程shell命令提取行

2、tensorflow fashion mnist => train.py 使用Wide resnet+SeNet, 结果应该在0.95左右，如果加上random erase（见transform.py）应该可以到0.96