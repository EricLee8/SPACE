lang=python #programming language
lr=1e-4
batch_size=64
beam_size=10
source_length=240
data_flow_length=60
target_length=30
data_dir=data
output_dir=output/$2
train_file=$data_dir/$lang/train/
dev_file=$data_dir/$lang/dev/
test_file=$data_dir/$lang/test/
epochs=20
pretrained_model=microsoft/graphcodebert-base
adv_lr=5e-5
adv_steps=3
entity_adv=1

CUDA_VISIBLE_DEVICES=$1 python run_adv.py \
    --model_type roberta \
    --model_name_or_path $pretrained_model \
    --train_filename $train_file \
    --dev_filename $dev_file \
    --test_filename $test_file \
    --output_dir $output_dir \
    --max_source_length $source_length \
    --max_target_length $target_length \
    --data_flow_length $data_flow_length \
    --beam_size $beam_size \
    --train_batch_size $batch_size \
    --eval_batch_size $batch_size \
    --learning_rate $lr \
    --num_train_epochs $epochs \
    --adv_lr $adv_lr \
    --adv_steps $adv_steps \
    --entity_adv $entity_adv \
    --lang $lang \
    --fp16 1 \
    --do_train \
    --do_eval \
    --do_test 
