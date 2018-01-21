#!/usr/bin/env bash

CODE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

export PYTHONPATH=$PYTHONPATH:$CODE_DIR
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR

pip install -r $CODE_DIR/requirements.txt

# create all the necessary directories
DATA_DIR=data
DOWNLOAD_DIR=downloads
PLOTS_DIR=history_plots
LOGS_DIR=logs
MODEL_CKPTS_DIR=model_ckpts
TEST_LOGS_DIR=../tests/logs

mkdir -p $DATA_DIR
mkdir -p $DOWNLOAD_DIR
mkdir -p $PLOTS_DIR
mkdir -p $LOGS_DIR
mkdir -p $MODEL_CKPTS_DIR
mkdir -p $TEST_LOGS_DIR

# script to download and preprocess the dataset (mainly splitting into train, validation and test sets)
python2 $CODE_DIR/data_preprocess.py
# script to train the model
python2 $CODE_DIR/train.py

