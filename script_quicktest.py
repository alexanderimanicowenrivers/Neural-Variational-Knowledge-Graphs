"""
            used for cluster job submission and parameter search over main2.py, final model analysis

                """

import itertools
import os

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
    #     command = 'PYTHONPATH=. python3-gpu {}/main.py {} ' \
    #     '--file_name {} ' \ this is for command if I want tensorboard

    command = 'PYTHONPATH=. anaconda-python3-cpu {}/main2.py  ' \
              '--mean_c {} ' \
              '--init_sig {} ' \
              '--embedding_size {} ' \
              '--alt_cost {} ' \
              '--Sigma_alt {} ' \
              '--alternating_updates {} ' \
              '--alt_opt {} ' \
              '--margin {} ' \
              '--file_name {} ' \
              '--decay_kl {}' \
        .format(path,
                #                 params,
                #                 set_to_path[c['instances']],
                c['w1'],
                c['w2'],
                c['w3'],
                c['w4'],
                c['w5'],
                c['w6'],
                c['w7'],
                c['w8'],
                "{}/logs/18_6_25_hinge/tb_nvkg.{}".format(path, summary(c)),
                c['w9'],
                )
    return command


def to_logfile(c, path):
    outfile = "%s/uclcs_nvkg_v1.%s.log" % (path, summary(c))
    return outfile


def main(_):
    hyperparameters_space = dict(
        w1=[5],
        w2=[0.5], #
        w3=[200],
        w4=[True],
        w5=[True],
        w6 = [False],
        w7=[True],
        w8=[7],
        w9=[False],

    )

    configurations = cartesian_product(hyperparameters_space)

    path = '/home/acowenri/workspace/Neural-Variational-Knowledge-Graphs/logs/18_6_25_hinge'

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

# use #$ -pe smp 1000 for 1000 cores
 # add this in for GPU's   # $ -P gpu
 #    # $ -l gpu=0

    header = """#!/bin/bash

#$ -cwd
#$ -S /bin/bash
#$ -o /home/acowenri/array.o.log
#$ -e /home/acowenri/array.e.log
#$ -t 1-{}
#$ -l tmem=8G
#$ -l h_rt=12:00:00
#$ -ac allow=LMNOPQSTU



export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd /home/acowenri/workspace/Neural-Variational-Knowledge-Graphs
export PYTHONPATH=.

    """.format(nb_jobs)

    print(header)

    #repeat each job three times

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && {}'.format(job_id, command_line))
    for job_id2, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && {}'.format(job_id+job_id2, command_line))
    for job_id3, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && {}'.format(job_id+job_id2+job_id3, command_line))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])