# MAC pretrain
torchrun --nproc_per_node=8 --master_port=22447 --max_restarts=0 pretrain.py \
  --model_name microsoft/Phi-3.5-vision-instruct \
  --output_dir runs/ph3v_pretrain_bs_256_low_expand_vision_text \
  --bf16 --pooling last \
  --lora \
  --lora_r 8 \
  --image_dir ./data/MMEB-train \
  --max_len 4096 \
  --logging_steps 1 \
  --lr_scheduler_type linear --learning_rate 2e-5 \
  --warmup_steps 200 --save_steps 500 --normalize True \
  --temperature 0.02 --per_device_train_batch_size 32 \
  --save_safetensors False --remove_unused_columns False \
  --report_to tensorboard \
  --image_resolution low \
  --seed 42 \
  --gradient_checkpointing true \
  --pretrain_use_vision_expand_loss \
  --pretrain_use_text_expand_loss \


# VEIN finetune
# we use 32*h20 for fine-tune, if the resource is limited, please try decrease the batch size
deepspeed --master_addr $MASTER_ADDR --hostfile /etc/mpi/hostfile train.py \
  --model_name microsoft/Phi-3.5-vision-instruct \
  --output_dir runs/phi3v_bs_128_no_flash_lr_2e-5_rank_8_ph3v_pretrain_bs_256_low_expand_vision_text_1500_vein_0.2_-1_gt_pos \
  --bf16 --pooling last \
  --lora \
  --lora_r 8 \
  --dataset_name TIGER-Lab/MMEB-train \
  --split_name original \
  --subset_name ImageNet_1K N24News HatefulMemes VOC2007 SUN397 OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA Visual7W VisDial CIRR VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t NIGHTS WebQA MSCOCO \
  --image_dir ./data/MMEB-train \
  --logging_steps 1 \
  --lr_scheduler_type linear --learning_rate 2e-5 \
  --warmup_steps 200 --save_steps 2000 --normalize True \
  --temperature 0.02 --per_device_train_batch_size 4 \
  --save_safetensors False --remove_unused_columns False \
  --report_to tensorboard \
  --image_resolution "low" \
  --not_use_flash_attn \
  --gradient_checkpointing true \
  --pretrain_dir runs/ph3v_pretrain_bs_256_low_expand_vision_text/checkpoint-1500 \
  --use_vein \
  --vein_mask_ratio 0.2 \
  --vein_layer_to_apply -1 \
  --vein_on_only_pure_vision_task \