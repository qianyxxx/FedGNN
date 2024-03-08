# main.py

import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from federated_runner import run_federated_learning

parser = argparse.ArgumentParser(description='Run Federated GNN Learning')
parser.add_argument('--num_rounds', type=int, default=20, help='Number of federated learning rounds')
parser.add_argument('--num_clients', type=int, default=5, help='Number of clients')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--dataset_path', type=str, default='path/to/your/dataset', help='Path to the dataset')
parser.add_argument('--dataset_name', type=str, default='Cora', help='Name of the dataset')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs per training round')

args = parser.parse_args()

# 运行部分和之前一样，传入 epochs 参数
run_federated_learning(args.num_rounds, args.num_clients, args.batch_size, args.dataset_path, args.dataset_name, args.epochs)