#!/usr/bin/env bash

CODE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

export PYTHONPATH=$PYTHONPATH:$CODE_DIR
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR

pip install -r $CODE_DIR/requirements.txt

# script to download and preprocess the dataset (mainly splitting into train, validation and test sets)
python2 $CODE_DIR/dataset_preprocess.py
# script to train the model
python2 $CODE_DIR/utils/train.py

