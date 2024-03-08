#!/bin/bash

# 定义超参数
NUM_ROUNDS=20
NUM_CLIENTS=10
BATCH_SIZE=32
DATASET_PATH="path/to/your/dataset"
DATASET_NAME="Cora"
EPOCHS=50

# 执行 main.py 脚本
/opt/homebrew/bin/python3 "main.py" \
  --num_rounds $NUM_ROUNDS \
  --num_clients $NUM_CLIENTS \
  --batch_size $BATCH_SIZE \
  --dataset_path $DATASET_PATH \
  --dataset_name $DATASET_NAME \
  --epochs $EPOCHS
