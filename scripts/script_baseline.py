"""
            used for cluster job submission and parameter search over baseline_main.py for baseline results

                """

import itertools
import os
import numpy as np
import sys
import logging


def cartesian_product(dicts):
    return list(dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs])


def to_cmd(c):
    #     set_to_path = {
    #         'train': 'data/wn18/snli_1.0_train.jsonl.gz'
    #     }

    path = '/home/acowenri/workspace/Neural-Variational-Knowledge-Graphs'
    #     params = '-m cbilstm -b 32 -d 0.8 -r 300 -o adam --lr 0.001 -c 100 -e 10 ' \
    #              '--restore saved/snli/cbilstm/2/cbilstm -C 5000'
    #     command = 'PYTHONPATH=. python3-gpu {}/main3.py {} ' \
    #     '--file_name {} ' \ this is for command if I want tensorboard

    command = 'PYTHONPATH=. anaconda-python3-cpu {}/baseline_main.py  ' \
              '--embedding_size {} ' \
              '--dataset {} ' \
              '--epsilon {} ' \
              '--lr {} ' \
              '--score_func {} ' \
              '--no_batches {} ' \
        .format(path,
                #                 params,
                #                 set_to_path[c['instances']],
                c['w3'],
                c['w6'],
                c['w7'],
                c['w8'],
                c['w9'],
                c['w10']
                )
    return command


def to_logfile(c, path):
    outfile = "%s/uclcs_nvkg_v1.%s.log" % (path, summary(c))
    return outfile


def main(_):
    hyperparameters_space = dict(
        w3=list(np.arange(49)+1),
        w6=['wn18rr' ],
        w7=[1e-7],
        w8=[0.001],
        w9=['DistMult'],
        w10=[100]
    )

    configurations = cartesian_product(hyperparameters_space)

    path = '/home/acowenri/workspace/Neural-Variational-Knowledge-Graphs/logs/low_rank_baseline'

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/acowenri/'):
        # If the folder that will contain logs does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

    configurations = list(configurations)

    command_lines = set()
    for cfg in configurations:
        logfile = to_logfile(cfg, path)

        completed = False
        if os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = 'Training finished' in content

        if not completed:
            command_line = '{} >> {} 2>&1'.format(to_cmd(cfg), logfile)
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines,reverse=True)
    nb_jobs = len(sorted_command_lines)


 # add this in for GPU's   # $ -P gpu
 #    # $ -l gpu=0

    header = """#!/bin/bash

#$ -cwd
#$ -S /bin/bash
#$ -o /dev/null
#$ -e /dev/null
#$ -t 1-{}
#$ -l tmem=12G
#$ -l h_rt=24:00:00
#$ -ac allow=LMNOPQSTU

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd /home/acowenri/workspace/Neural-Variational-Knowledge-Graphs
export PYTHONPATH=.

    """.format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && {}'.format(job_id, command_line))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])