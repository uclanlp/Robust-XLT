#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
GPU=${2:-0}
DATA_DIR=${3:-"$REPO/data_generalized_augment/"}
OUT_DIR=${4:-"$REPO/outputs/"}
MODEL_DIR=${5}

export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=8

TASK='pawsx'
LR=2e-5
EPOCH=3
MAXL=128
LANGS='en-en,en-es,en-de,en-fr,en-zh,en-ko,en-ja,es-en,es-es,es-de,es-fr,es-zh,es-ko,es-ja,de-en,de-es,de-de,de-fr,de-zh,de-ko,de-ja,fr-en,fr-es,fr-de,fr-fr,fr-zh,fr-ko,fr-ja,zh-en,zh-es,zh-de,zh-fr,zh-zh,zh-ko,zh-ja,ko-en,ko-es,ko-de,ko-fr,ko-zh,ko-ko,ko-ja,ja-en,ja-es,ja-de,ja-fr,ja-zh,ja-ko,ja-ja'
LC=""
if [ $MODEL == "bert-base-multilingual-cased" ] || [ $MODEL == "bert-base-multilingual-uncased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
fi

if [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=2
  GRAD_ACC=16
else
  BATCH_SIZE=8
  GRAD_ACC=4
fi

SAVE_DIR="${OUT_DIR}/${TASK}/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}/"
mkdir -p $SAVE_DIR

python $PWD/third_party/run_classify.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL \
  --train_language en \
  --task_name $TASK \
  --do_predict \
  --train_split train \
  --test_split test \
  --data_dir $DATA_DIR/$TASK \
  --gradient_accumulation_steps $GRAD_ACC \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --max_seq_length $MAXL \
  --output_dir $SAVE_DIR \
  --eval_all_checkpoints \
  --overwrite_output_dir \
  --overwrite_cache \
  --save_steps 500 \
  --log_file 'train.log' \
  --predict_languages $LANGS \
  --save_only_best_checkpoint \
  --eval_test_set $LC \
  --init_checkpoint $MODEL_DIR
