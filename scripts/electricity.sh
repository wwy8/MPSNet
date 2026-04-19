##!/bin/bash
#SBATCH -p bingxing
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#module load apps/Miniforge3/25.3.1-0
#source activate pygpu
#export PYTHONUNBUFFERED=1


if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=HARMON

root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

seq_len=720
#for seq_len in 48 96 192 336 528
#do
for random_seed in 3405 3406 3408 3409
do
  for pred_len in 96 192 336 720
  do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --period_len 24 \
      --model_type 'mlp' \
      --d_model 128 \
      --enc_in 321 \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 128 --learning_rate 0.02 --random_seed $random_seed
  done
done
