codebert="0" # use "1" to train CodeBERT and "0" to train GraphCodeBERT
cuda="0"
save_name="space"
entity_adv="1"
adv_steps="3"
adv_lr="5e-4"

python run_adv.py \
    --do_train \
    --train_data_file=dataset/train.jsonl \
    --eval_data_file=dataset/valid.jsonl \
    --test_data_file=dataset/test.jsonl \
    --epoch 5 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --cuda $cuda \
    --save_name $save_name \
    --entity_adv $entity_adv \
    --adv_steps $adv_steps \
    --adv_lr $adv_lr \
    --codebert $codebert \
    --seed 123456