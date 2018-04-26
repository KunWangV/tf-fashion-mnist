# coding: utf-8
import random
import os
import time
from argparse import ArgumentParser
from rn.rn import sampler


class Timer:
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.time0 = time.time()
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        time1 = time.time()
        print("-- {:20}: cost time {}s --".format(self.name, time1 - self.time0))


def count_line(filename):
    """
    计算文件中一共有多少行
    :param filename:
    :return:
    """
    cmd = "sed -n '$=' {}".format(filename)
    with Timer('count line'):
        file = os.popen(cmd)
        n = int(file.read())

    return n


def gen_numbers(n, k=1000):
    """
    蓄水池抽样 生成k个随机数
    :param n: 总共n个数
    :param k: k个样本
    :return:
    """
    pool = list(range(1, k + 1))
    choice_idx = list(range(0, k))

    with Timer('get random numbers'):
        for i in range(k + 1, n + 1):
            if random.random() < k / float(i):
                pool[random.choice(choice_idx)] = i

    return pool


def get_lines(filename, numbers):
    """
    文件中提取k行
    :param filename: 文件名
    :param numbers: 集合
    :return:
    """

    if len(numbers) == 0:
        return

    pt = "sed '{};d' {}"
    lines = []
    numbers = sorted(numbers)
    for n in numbers:
        lines.append('{}p'.format(n))

    lines[-1].replace('p', 'q')

    cmd = pt.format(';'.join(lines), filename)
    with Timer('get select lines'):
        file = os.popen(cmd)
        ans = file.read()
        print(ans)

    return ans


def reader_pool(args, numbers):
    from threading import Thread
    import queue

    numbers = sorted(numbers)
    n_bin = len(numbers) // args.cc
    start = 0
    end = 0
    q = queue.Queue()
    for i in range(args.cc):
        end = (i + 1) * n_bin
        sub = numbers[start:end]
        start = end
        t = Thread(target=get_lines, args=(args.filename, sub,))
        t.start()
        q.put(t)

    if end < len(numbers):
        sub = numbers[end:len(numbers)]
        t = Thread(target=get_lines, args=(args.filename, sub,))
        t.start()
        q.put(t)

    q.join()


def main(args):
    if args.ops == 'e':
        if args.n is None:
            n = count_line(args.filename)
        else:
            n = args.n

        with Timer('generate random numbers'):
            numbers = sampler.gen_numbers(n, args.k)
            assert len(numbers) == args.k

        reader_pool(args, numbers)

    if args.ops == 'c':
        n = count_line(args.filename)
        print(n)


if __name__ == '__main__':
    parse = ArgumentParser()
    parse.add_argument('--filename', required=True)
    parse.add_argument('--ops', default='e', help='operatins: e-random select lines, c- get total number of lines')
    parse.add_argument('--n', type=int, default=None, help='set if you known total number of lines')
    parse.add_argument('--k', type=int, default=1000, help='number of lines to select randomly')
    parse.add_argument('--cc', type=int, default=4, help='concurrency of extract lines')

    args = parse.parse_args()
    main(args)
