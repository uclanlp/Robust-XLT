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

TASK='xnli'
LR=2e-5
EPOCH=1
MAXL=128
LANGS='en-en,en-es,en-de,en-fr,en-bg,en-ru,en-el,en-th,en-sw,en-vi,en-ar,en-zh,en-hi,en-ur,en-tr,es-en,es-es,es-de,es-fr,es-bg,es-ru,es-el,es-th,es-sw,es-vi,es-ar,es-zh,es-hi,es-ur,es-tr,de-en,de-es,de-de,de-fr,de-bg,de-ru,de-el,de-th,de-sw,de-vi,de-ar,de-zh,de-hi,de-ur,de-tr,fr-en,fr-es,fr-de,fr-fr,fr-bg,fr-ru,fr-el,fr-th,fr-sw,fr-vi,fr-ar,fr-zh,fr-hi,fr-ur,fr-tr,bg-en,bg-es,bg-de,bg-fr,bg-bg,bg-ru,bg-el,bg-th,bg-sw,bg-vi,bg-ar,bg-zh,bg-hi,bg-ur,bg-tr,ru-en,ru-es,ru-de,ru-fr,ru-bg,ru-ru,ru-el,ru-th,ru-sw,ru-vi,ru-ar,ru-zh,ru-hi,ru-ur,ru-tr,el-en,el-es,el-de,el-fr,el-bg,el-ru,el-el,el-th,el-sw,el-vi,el-ar,el-zh,el-hi,el-ur,el-tr,th-en,th-es,th-de,th-fr,th-bg,th-ru,th-el,th-th,th-sw,th-vi,th-ar,th-zh,th-hi,th-ur,th-tr,sw-en,sw-es,sw-de,sw-fr,sw-bg,sw-ru,sw-el,sw-th,sw-sw,sw-vi,sw-ar,sw-zh,sw-hi,sw-ur,sw-tr,vi-en,vi-es,vi-de,vi-fr,vi-bg,vi-ru,vi-el,vi-th,vi-sw,vi-vi,vi-ar,vi-zh,vi-hi,vi-ur,vi-tr,ar-en,ar-es,ar-de,ar-fr,ar-bg,ar-ru,ar-el,ar-th,ar-sw,ar-vi,ar-ar,ar-zh,ar-hi,ar-ur,ar-tr,zh-en,zh-es,zh-de,zh-fr,zh-bg,zh-ru,zh-el,zh-th,zh-sw,zh-vi,zh-ar,zh-zh,zh-hi,zh-ur,zh-tr,hi-en,hi-es,hi-de,hi-fr,hi-bg,hi-ru,hi-el,hi-th,hi-sw,hi-vi,hi-ar,hi-zh,hi-hi,hi-ur,hi-tr,ur-en,ur-es,ur-de,ur-fr,ur-bg,ur-ru,ur-el,ur-th,ur-sw,ur-vi,ur-ar,ur-zh,ur-hi,ur-ur,ur-tr,tr-en,tr-es,tr-de,tr-fr,tr-bg,tr-ru,tr-el,tr-th,tr-sw,tr-vi,tr-ar,tr-zh,tr-hi,tr-ur,tr-tr'
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
