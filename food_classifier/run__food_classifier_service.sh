#!/usr/bin/env bash

CODE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

export PYTHONPATH=$PYTHONPATH:$CODE_DIR
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR

# script to serve the model as a flask rest api
python2 $CODE_DIR/utils/api.py
