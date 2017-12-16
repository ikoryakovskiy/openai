from __future__ import division
import multiprocessing
import os
import os.path
from time import sleep
from datetime import datetime
import collections
import argparse
import itertools
import signal
import random

counter_lock = multiprocessing.Lock()
cores = 0
random.seed(datetime.now())

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument('-c', '--cores', type=int, help='specify maximum number of cores')
    args = parser.parse_args()
    if args.cores:
        args.cores = min(multiprocessing.cpu_count(), args.cores)
    else:
        args.cores = min(multiprocessing.cpu_count(), 32)
    print('Using {} cores.'.format(args.cores))

    # Parameters
    runs = range(10)

    ###
    nb_epochs = [25]
    options = []
    for r in itertools.product(nb_epochs, runs): options.append(r)
    options = [flatten(tupl) for tupl in options]

    configs = [
                "cfg/rbdl_py_balancing.yaml",
              ]
    L1 = rl_run_zero_shot(args, configs, options)


    ###
    nb_epochs = [150]
    options = []
    for r in itertools.product(nb_epochs, runs): options.append(r)
    options = [flatten(tupl) for tupl in options]

    configs = [
                "cfg/rbdl_py_walking.yaml",
              ]
    L2 = rl_run_zero_shot(args, configs, options)

    L = L1+L2
    random.shuffle(L)
    do_multiprocessing_pool(args, L)


######################################################################################
def rl_run_zero_shot(args, list_of_cfgs, options):
    list_of_new_cfgs = []

    loc = "tmp"
    if not os.path.exists(loc):
        os.makedirs(loc)

    for cfg in list_of_cfgs:
        fname, fext = os.path.splitext( cfg.replace("/", "_") )

        for o in options:
            str_o = "-".join(map(lambda x : "{:06d}".format(int(round(100000*x))), o[:-1]))  # last element in 'o' is reserved for mp
            if not str_o:
                str_o += "mp{}".format(o[-1])
            else:
                str_o += "-mp{}".format(o[-1])
            print("Generating parameters: {}".format(str_o))

            # create local filename
            list_of_new_cfgs.append( "{}/{}-{}.cfg".format(loc, fname, str_o) )

            nb_epochs = o[0]
            output = "{}-{}".format(fname, str_o)

            cmd = 'python3 ddpg.py --cfg={} --nb-epochs={} --output={}'.format(cfg, nb_epochs, output)
            
            with open(list_of_new_cfgs[-1], 'w') as file:
                file.write(cmd)

    print(list_of_new_cfgs)

    return list_of_new_cfgs


######################################################################################
def mp_run(cfg):
    # Multiple copies can be run on one computer at the same time, which results in the same seed for a random generator.
    # Thus we need to wait for a second or so between runs
    global counter
    global cores
    with counter_lock:
        wait = counter.value
        counter.value += 2
    sleep(wait)
    print('wait finished {0}'.format(wait))
    # Run the experiment
    with open(cfg, 'r') as file:
        cmd = file.read()
    os.system(cmd)


######################################################################################
def init(cnt, num):
    """ store the counter for later use """
    global counter
    global cores
    counter = cnt
    cores = num


######################################################################################
def do_multiprocessing_pool(args, list_of_new_cfgs):
    """Do multiprocesing"""
    counter = multiprocessing.Value('i', 0)
    cores = multiprocessing.Value('i', args.cores)
    print('cores {0}'.format(cores.value))
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = multiprocessing.Pool(args.cores, initializer = init, initargs = (counter, cores))
    signal.signal(signal.SIGINT, original_sigint_handler)
    try:
        pool.map(mp_run, list_of_new_cfgs)
    except KeyboardInterrupt:
        pool.terminate()
    else:
        pool.close()
    pool.join()
######################################################################################


if __name__ == "__main__":
    main()

