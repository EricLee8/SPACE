cuda="0"
save_name="bsl"

python3 run.py \
    --lang=$1 \
    --do_train \
    --train_data_file=dataset/$1/train.jsonl \
    --eval_data_file=dataset/$1/valid.jsonl \
    --test_data_file=dataset/$1/test.jsonl \
    --codebase_file=dataset/$1/codebase.jsonl \
    --num_train_epochs 10 \
    --learning_rate 2e-5 \
    --code_length 320 \
    --data_flow_length 64 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --cuda $cuda \
    --save_name $save_name \
    --seed 123456 \
    --codebert $2
