codebert="0" # use "1" to train CodeBERT and "0" to train GraphCodeBERT
cuda="0"
save_name="baseline"
lr="2e-5"

python run.py \
    --do_train \
    --train_data_file=dataset/train.jsonl \
    --eval_data_file=dataset/valid.jsonl \
    --test_data_file=dataset/test.jsonl \
    --epoch 5 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate $lr \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --cuda $cuda \
    --save_name $save_name \
    --seed 123456 \
    --codebert $codebert
