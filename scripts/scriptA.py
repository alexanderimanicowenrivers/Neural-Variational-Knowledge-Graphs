"""
            used for cluster job submission and parameter search over main.py, train variational knowledge graph embeddings

                """


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    #     command = 'PYTHONPATH=. python3-gpu {}/baseline_main.py {} ' \
    #     '--file_name {} ' \ this is for command if I want tensorboard

    command = 'PYTHONPATH=. anaconda-python3-cpu {}/mainA.py  ' \
              '--no_batches {} ' \
              '--epsilon {} ' \
              '--embedding_size {} ' \
              '--dataset {} ' \
              '--alt_opt {} ' \
              '--lr {} ' \
              '--score_func {} ' \
              '--negsamples {} ' \
              '--projection {} ' \
              '--distribution {} ' \
              '--ablation {} ' \
              '--file_name {} ' \
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
                c['w9'],
                c['w10'],
                c['w11'],
                ("{}/logs/low_rank_ModelA/uclcs_nvkg_v1.{}".format(path, summary(c)))
                )
    return command


def to_logfile(c, path):
    outfile = "%s/uclcs_nvkg_v1.%s.log" % (path, summary(c))
    return outfile


def main(_):
    hyperparameters_space = dict(
        w1=[2,5,10,20,30,40,50],
        w2=[1e-7,1e-3], #ls
        w3=[5],
        w4=['fb15k-237', 'kinship', 'nations', 'umls', 'wn18', 'wn18rr'],
        w5=[False],
        w6=[0.001,0.0001],
        w7=['DistMult','ComplEx'],
        w8=[5],
        w9=[False],
        w10=['normal','vmf'],
        w11=[0])

    configurations = cartesian_product(hyperparameters_space)
    path = '/home/acowenri/workspace/Neural-Variational-Knowledge-Graphs/logs/low_rank_ModelA'


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
    sorted_command_lines = sorted(command_lines)
    nb_jobs = len(sorted_command_lines)

    header = """#!/bin/bash

    # $ -cwd
    # $ -S /bin/bash
    # $ -o /dev/null
    # $ -e /dev/null
    # $ -t 1-{}
    # $ -l tmem=12G
    # $ -l h_rt=24:00:00
    # $ -P gpu
    # $ -l gpu=1-GPU_PASCAL=1

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





#
#
# import itertools
# import os
#
# import sys
# import logging
#
#
# def cartesian_product(dicts):
#     return list(dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))
#
#
# def summary(configuration):
#     kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
#     return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs])
#
#
# def to_cmd(c):
#     #     set_to_path = {
#     #         'train': 'data/wn18/snli_1.0_train.jsonl.gz'
#     #     }
#
#     path = '/home/acowenri/workspace/Neural-Variational-Knowledge-Graphs'
#     #     params = '-m cbilstm -b 32 -d 0.8 -r 300 -o adam --lr 0.001 -c 100 -e 10 ' \
#     #              '--restore saved/snli/cbilstm/2/cbilstm -C 5000'
#     #     command = 'PYTHONPATH=. python3-gpu {}/baseline_main.py {} ' \
#     #     '--file_name {} ' \ this is for command if I want tensorboard
#
#     command = 'PYTHONPATH=. anaconda-python3-cpu {}/main.py  ' \
#               '--no_batches {} ' \
#               '--epsilon {} ' \
#               '--embedding_size {} ' \
#               '--dataset {} ' \
#               '--alt_updates {} ' \
#               '--lr {} ' \
#               '--score_func {} ' \
#               '--negsamples {} ' \
#               '--alt_test {} ' \
#               '--file_name {} ' \
#         .format(path,
#                 #                 params,
#                 #                 set_to_path[c['instances']],
#                 c['w1'],
#                 c['w2'],
#                 c['w3'],
#                 c['w4'],
#                 c['w5'],
#                 c['w6'],
#                 c['w7'],
#                 c['w8'],
#                 c['w9'],
#                 ("{}/logs/180807_scaled/uclcs_nvkg_v1.{}".format(path, summary(c)))
#                 )
#     return command
#
#
# def to_logfile(c, path):
#     outfile = "%s/uclcs_nvkg_v1.%s.log" % (path, summary(c))
#     return outfile
#
#
# def main(_):
#     hyperparameters_space = dict(
#         w1=[1000],
#         # w1=[10],
#         w2=[1e-3], #
#         w3=[200,300],
#         w4 = ['kinship','nations','umls'],
#         w5=[False],
#         w6=[0.01,0.001],
#         w7=['DistMult','TransE', 'ComplEx'],
#         w8=[5,10,15,20,25],
#         w9=['none'])
#
#     configurations = cartesian_product(hyperparameters_space)
#
#     path = '/home/acowenri/workspace/Neural-Variational-Knowledge-Graphs/logs/180807_scaled'
#
#     # Check that we are on the UCLCS cluster first
#     if os.path.exists('/home/acowenri/'):
#         # If the folder that will contain logs does not exist, create it
#         if not os.path.exists(path):
#             os.makedirs(path)
#
#     configurations = list(configurations)
#
#     command_lines = set()
#     for cfg in configurations:
#         logfile = to_logfile(cfg, path)
#
#         completed = False
#         if os.path.isfile(logfile):
#             with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
#                 content = f.read()
#                 completed = 'Training finished' in content
#
#         if not completed:
#             command_line = '{} >> {} 2>&1'.format(to_cmd(cfg), logfile)
#             command_lines |= {command_line}
#
#     # Sort command lines and remove duplicates
#     sorted_command_lines = sorted(command_lines,reverse=True)
#     nb_jobs = len(sorted_command_lines)
#
# # use #$ -pe smp 1000 for 1000 cores
#  # add this in for GPU's   # $ -P gpu
#  #    # $ -l gpu=0
#     #no gpu
#
# $ -cwd
# $ -S /bin/bash
# $ -o /dev/null
# $ -e /dev/null
# $ -t 1-{}
# $ -l tmem=12G
# $ -l h_rt=48:00:00
# $ -ac allow=LMNOPQSTU
#
#     #GPU
#
#     # $ -cwd
#     # $ -S /bin/bash
#     # $ -o /dev/null
#     # $ -e /dev/null
#     # $ -t 1-{}
#     # $ -l tmem=8G
#     # $ -l h_rt=24:00:00
#     # $ -P gpu
#     # $ -l gpu=1-GPU_PASCAL=1
#
#
#     header = """#!/bin/bash
#
# # $ -cwd
# # $ -S /bin/bash
# # $ -o /home/acowenri/array.o.log
# # $ -e /home/acowenri/array.e.log
# # $ -t 1-{}
# # $ -l tmem=8G
# # $ -l h_rt=12:00:00
# # $ -ac allow=LMNOPQSTU
#
# export LANG="en_US.utf8"
# export LANGUAGE="en_US:en"export LANGUAGE="en_US:en"
#
# cd /home/acowenri/workspace/Neural-Variational-Knowledge-Graphs
# export PYTHONPATH=.
#
#     """.format(nb_jobs)
#
#     print(header)
#
#     #repeat each job three times
#
#     for job_id, command_line in enumerate(sorted_command_lines, 1):
#         print('test $SGE_TASK_ID -eq {} && {}'.format(job_id, command_line))
#
#
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     main(sys.argv[1:])