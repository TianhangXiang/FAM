import os
import random
import logging
import sys
import torch
import wandb
import numpy as np
from transformers import (
    HfArgumentParser,
)
from transformers import logging as hf_logging
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model import MMEBModel, VEINMMEBModel
from src.trainer import MMEBTrainer, VEINTrainer
from src.utils import print_rank
from src.model_utils import load_processor, get_backbone_name
from src.model_utils import print_master
from src.dataset import TrainJsonDataset
from src.collator import TrainCollator

def seed_all(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)
    
def setup_logging_to_file(log_file_path):
    logger = hf_logging.get_logger()  # 或者 logging.getLogger("transformers")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 你自己代码的 logger（比如 logger = logging.getLogger(__name__)）也建议添加这个 handler
    logging.getLogger().addHandler(file_handler)

def main():
    
    seed_all()
    # a hack for torch.distributed.launch: https://github.com/huggingface/transformers/issues/22171
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if 'wandb' in training_args.report_to:
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or (not torch.distributed.is_initialized()):
            print_rank('init wandb')
            wandb.init(project=training_args.project_name, name=training_args.run_name, mode="online")

    if training_args.local_rank in [-1, 0]:
        # 创建输出目录（如果不存在）
        os.makedirs(training_args.output_dir, exist_ok=True)

        # 设置日志文件路径
        log_file = os.path.join(training_args.output_dir, "training.log")
        # 初始化日志记录
        setup_logging_to_file(log_file)
    
    print_master(f"model_args: {model_args}")
    print_master(f"data_args: {data_args}")
    print_master(f"training_args: {training_args}")
    
    if training_args.use_vein:
        model = VEINMMEBModel.build(model_args, training_args)
    else:    
        model = MMEBModel.build(model_args, training_args)
    
    model_backbone = get_backbone_name(hf_config=model.config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_backbone}')
    processor = load_processor(model_args)
    setattr(model, 'processor', processor)

    train_dataset = TrainJsonDataset(data_args, model_args)    
    collator = TrainCollator(data_args, model_args, processor)
    
    if training_args.use_vein:
       trainer = VEINTrainer(
            model=model,
            processing_class=processor,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collator,
       )
    else:
        trainer = MMEBTrainer(
            model=model,
            processing_class=processor,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collator,
        )
    train_dataset.trainer = trainer

    trainer.train()
    trainer.save_model(training_args.output_dir)

    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
