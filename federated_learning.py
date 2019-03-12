import json
import time
import os
import argparse
import random

import numpy as np
import torch, torchvision
import torch.optim as optim

import data_utils
import neural_nets
import distributed_training_utils as dst
from distributed_training_utils import Client, Server
import experiment_manager as xpm
import default_hyperparameters as dhp

import copy

parser = argparse.ArgumentParser()
parser.add_argument("--schedule", default="main", type=str)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=None, type=int)
parser.add_argument("--reverse_order", default=False, type=bool)

print("Torch Version: ", torch.__version__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = parser.parse_args()

# Load the Hyperparameters of all Experiments to be performed and set up the Experiments
with open('federated_learning.json') as data_file:    
  experiments_raw = json.load(data_file)[args.schedule]

hp_dicts = [hp for x in experiments_raw for hp in xpm.get_all_hp_combinations(x)][args.start:args.end]
if args.reverse_order:
  hp_dicts = hp_dicts[::-1]
experiments = [xpm.Experiment(hyperparameters=hp) for hp in hp_dicts]


def run_experiments(experiments):
  print("Running {} Experiments..\n".format(len(experiments)))
  for xp_count, xp in enumerate(experiments):
    hp = dhp.get_hp(xp.hyperparameters)
    xp.prepare(hp)
    print(xp)

    # Load the Data and split it among the Clients
    client_loaders, train_loader, test_loader, stats = data_utils.get_data_loaders(hp)

    # Instantiate Clients and Server with Neural Net
    net = getattr(neural_nets, hp['net'])
    clients = [Client(loader, net().to(device), hp, xp, id_num=i) for i, loader in enumerate(client_loaders)]
    server = Server(test_loader, net().to(device), hp, xp, stats)

    # Print optimizer specs
    print_model(device=clients[0])
    print_optimizer(device=clients[0])

    # Start Distributed Training Process
    print("Start Distributed Training..\n")
    t1 = time.time()

    for c_round in range(1, hp['communication_rounds']+1):

      participating_clients = random.sample(clients, int(len(clients)*hp['participation_rate'])) 

      # Clients do
      for client in participating_clients:
        client.synchronize_with_server(server)
        client.compute_weight_update(hp['local_iterations'])
        client.compress_weight_update_up(compression=hp['compression_up'], accumulate=hp['accumulation_up'], count_bits=hp["count_bits"])

      # Server does  
      server.aggregate_weight_updates(participating_clients, aggregation=hp['aggregation'])  
      server.compress_weight_update_down(compression=hp['compression_down'], accumulate=hp['accumulation_down'], count_bits=hp["count_bits"])

      # Evaluate  
      if xp.is_log_round(c_round):
        print("Experiment: {} ({}/{})".format(args.schedule, xp_count+1, len(experiments)))
        print("Evaluate...")
        results_train = server.evaluate(max_samples=5000, loader=train_loader)
        results_test = server.evaluate(max_samples=10000)
        
        # Logging
        xp.log({'communication_round' : c_round, 'lr' : clients[0].optimizer.__dict__['param_groups'][0]['lr'],
          'epoch' : clients[0].epoch, 'iteration' : c_round*hp['local_iterations']})
        xp.log({'client{}_loss'.format(client.id) : client.train_loss for client in clients}, printout=False)
        
        xp.log({key+'_train' : value for key, value in results_train.items()})
        xp.log({key+'_test' : value for key, value in results_test.items()})

        if hp["count_bits"]:
          xp.log({'bits_sent_up' : sum(participating_clients[0].bits_sent), 'bits_sent_down' : sum(server.bits_sent)}, printout=False)
        
        xp.log({'time' : time.time()-t1}, printout=False)

        # Save results to Disk
        if 'log_path' in hp and hp['log_path']:
          xp.save_to_disc(path=hp['log_path'])

        # Timing
        total_time = time.time()-t1
        avrg_time_per_c_round = (total_time)/c_round 
        e = int(avrg_time_per_c_round*(hp['communication_rounds']-c_round))
        print("Remaining Time (approx.):", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60), 
                  "[{:.2f}%]\n".format(c_round/hp['communication_rounds']*100))

    # Delete objects to free up GPU memory
    del server; clients.clear()
    torch.cuda.empty_cache()


def print_optimizer(device):
  try:
    print("Optimizer:", device.hp['optimizer'])
    for key, value in device.optimizer.__dict__['defaults'].items():
      print(" -", key,":", value)

    hp = device.hp
    base_batchsize = hp['batch_size']
    if hp['fix_batchsize']:
      client_batchsize = base_batchsize//hp['n_clients']
    else:
      client_batchsize = base_batchsize
    total_batchsize = client_batchsize*hp['n_clients']
    print(" - batchsize (/ total): {} (/ {})".format(client_batchsize, total_batchsize))
    print()
  except:
    pass


def print_model(device):
  print("Model {}:".format(device.hp['net']))
  n = 0
  for key, value in device.model.named_parameters():
    print(' -', '{:30}'.format(key), list(value.shape))
    n += value.numel()
  print("Total number of Parameters: ", n) 
  print()


if __name__ == "__main__":
  run_experiments(experiments)