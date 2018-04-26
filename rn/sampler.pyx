import random

def gen_numbers(n, k=1000):
    """
    蓄水池抽样 生成k个随机数
    :param n: 总共n个数
    :param k: k个样本
    :return:
    """
    pool = list(range(1, k + 1))
    choice_idx = list(range(0, k))
    for i in range(k + 1, n + 1):
        if random.random() < k / float(i):
            pool[random.choice(choice_idx)] = i

    return pool
