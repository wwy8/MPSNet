if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=MPSNet

root_path_name=./dataset/ETT/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

#seq_len=720
#for pred_len in 96 192 336 720
#do
seq_len=720
#pred_len=720
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
    --period_len 4 \
    --enc_in 7 \
    --model_type 'mlp' \
    --d_model 128 \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed $random_seed
  done
done
