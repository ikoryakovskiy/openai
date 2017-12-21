from __future__ import division
import multiprocessing
import os
import os.path
from time import sleep
from datetime import datetime
import collections
import itertools
import signal
import random
from ppo1 import parse_args, cfg_run
import yaml
import io

counter_lock = multiprocessing.Lock()
cores = 0
random.seed(datetime.now())

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def main():
    args = parse_args()
    
    if args['cores']:
        arg_cores = min(multiprocessing.cpu_count(), args['cores'])
    else:
        arg_cores = min(multiprocessing.cpu_count(), 32)
    print('Using {} cores.'.format(arg_cores))
    
    # Parameters
    runs = range(10)
    timesteps_per_actorbatch = [1024, 2048]

    alg = 'ppo1'
    
    ###
    num_timesteps = [200]
    options = []
    for r in itertools.product(timesteps_per_actorbatch, num_timesteps, runs): options.append(r)
    options = [flatten(tupl) for tupl in options]

    configs = [
                "cfg/rbdl_py_balancing_inf.yaml",
              ]
    L1 = rl_run_zero_shot(configs, alg, args, options)


    ###
    num_timesteps = [1000]
    options = []
    for r in itertools.product(timesteps_per_actorbatch, num_timesteps, runs): options.append(r)
    options = [flatten(tupl) for tupl in options]

    configs = [
                "cfg/rbdl_py_walking_inf.yaml",
              ]
    L2 = rl_run_zero_shot(configs, alg, args, options)

    L = L1+L2
    random.shuffle(L)
    do_multiprocessing_pool(arg_cores, L)


######################################################################################
def rl_run_zero_shot(list_of_cfgs, alg, args, options):
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
            list_of_new_cfgs.append( "{}/{}-{}-{}.yaml".format(loc, alg, fname, str_o) )

            args['cfg'] = cfg
            args['timesteps_per_actorbatch'] = o[0]
            args['num_timesteps'] = o[1]*1000
            args['output'] = "{}-{}-{}".format(alg, fname, str_o)
            args['save'] = True
                
            with io.open(list_of_new_cfgs[-1], 'w', encoding='utf8') as file:
                yaml.dump(args, file, default_flow_style=False, allow_unicode=True)

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
        args = yaml.load(file)
    cfg_run(**args)


######################################################################################
def init(cnt, num):
    """ store the counter for later use """
    global counter
    global cores
    counter = cnt
    cores = num


######################################################################################
def do_multiprocessing_pool(arg_cores, list_of_new_cfgs):
    """Do multiprocesing"""
    counter = multiprocessing.Value('i', 0)
    cores = multiprocessing.Value('i', arg_cores)
    print('cores {0}'.format(cores.value))
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = multiprocessing.Pool(arg_cores, initializer = init, initargs = (counter, cores))
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

