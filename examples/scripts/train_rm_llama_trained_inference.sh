set -x

read -r -d '' training_commands <<EOF
openrlhf.charles.test \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 1 \
   --micro_train_batch_size 1 \
   --pretrain ./checkpoint/llama-3-8b-rm-combined \
   --max_epochs 1 \
   --max_len 512 \
   --max_samples 10 \
   --zero_stage 2 \
   --learning_rate 9e-6 \
   --dataset shoubing35/test \
   --chosen_key chosen \
   --rejected_key rejected \
   --load_checkpoint \
   --gradient_checkpointing \
   --bf16 \
   --load_in_4bit \
   --lora_rank 0
EOF
     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
     # --packing_samples


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
