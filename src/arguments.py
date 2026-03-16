from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import List


@dataclass
class ModelArguments:
    model_name: str = field(
        metadata={"help": "huggingface model name or path"}
    )
    model_backbone: str = field(
        default=None,
        metadata={"help": "backbone name"}
    )
    processor_name: str = field(
        default=None, metadata={"help": "processor_name, huggingface model name or path"}
    )
    model_type: str = field(
        default=None, metadata={"help": "lavis model type"}
    )
    checkpoint_path: str = field(
        default=None, metadata={"help": "a local model path"}
    )
    pooling: str = field(
        default='last',
        metadata={"help": "pooling method for encoder"}
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "normalize query and passage representations"}
    )
    temperature: float = field(
        default=0.02,
        metadata={"help": "temperature for softmax"}
    )
    lora: bool = field(
        default=False, metadata={"help": "do parameter-efficient fine-tuning with lora"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "lora r"}
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "lora alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "lora dropout"}
    )
    lora_target_modules: str = field(
        default="qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj",
        metadata={"help": "lora target modules"}
    )
    num_crops: int = field(
        default=4,
        metadata={"help": "number of crops used in image encoder"}
    )
    
    vein_loss_type: str = field(
        default='cos',
        metadata={"help": "reconstructor loss type"}
    )
    vein_loss_weight: float = field(
        default=1.0,
    )
    vein_mask_ratio: float = field(
        default=0.1,
        metadata={"help": "mask ratio for reconstructor"}
    )

    vein_mask_dim: int = field(
        default=1536,
    )
    
    vein_mask_generate_type: str = field(
        default='sample',
        metadata={"help": "mask generate type"}
    )
    
    vein_detach_feature: bool = field(
        default=True,
        metadata={"help": "mask generate type"}
    )
    
    vein_add_pos_to_target: bool = field(
        default=True,
        metadata={"help": "mask generate type"}
    )

    vein_dropout: float = field(
        default=0.1,
        metadata={"help": "mask generate type"}
    )
    
    vein_layer_to_apply: int = field(
        default=-1,
    )
    
    vein_decoder_layers: int = field(
        default=1,
        metadata={"help": "reconstructor layers"}
    )
    
    vein_decoder_strategy: str = field(
        default=1,
        metadata={"help": "reconstructor strategy"}
    )
    
    vein_on_only_pure_vision_task: bool = field(
        default=True,
    )
    
    vein_on_retrieval_i2t_t2i_task: bool = field(
        default=True,
    )
        
    vein_norm_embedding: bool = field(
        default=False,
    )
    
    vein_use_projector: bool = field(
        default=False, metadata={"help": "use projector to mapping token to reconstructor"}
    )
    
    vein_pos_type: str = field(
        default='sin', 
    )
    
    not_use_flash_attn: bool = field(
        default=False,
    )
    
    pretrain_dir: str = field(
        default=None,
    )

    pretrain_nce_loss_weight: float = field(
        default=1.0,
    )
    
    pretrain_use_vision_expand_loss: bool = field(
        default=False,
    )
    
    pretrain_use_cross_loss: bool = field(
        default=False,
    )
    
    pretrain_use_filip_loss: bool = field(
        default=False,
    )
        
    pretrain_vision_loss_weight: float = field(
        default=1.0,
    )
    
    pretrain_use_text_expand_loss: bool = field(
        default=False,
    )
    
    pretrain_text_loss_weight: float = field(
        default=1.0,
    )
    
    cache_dir: str = field(
        default='',
    )
    
    temperature_learnable: bool = field(
        default=False,
    )
    
@dataclass
class DataArguments:
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    split_name: List[str] = field(
        default='original', metadata={"help": "'original', 'diverse_instruction'"}
    )
    subset_name: List[str] = field(
        default=None, metadata={"help": "Useful for datasets with subsets"}
    )
    dataset_split: str = field(
        default='train', metadata={"help": "dataset split"}
    )
    num_sample_per_subset: int = field(
        default=100, metadata={"help": "number of training samples per subset"}
    )
    image_dir: str = field(
        default=None, metadata={"help": "Image directory path"}
    )
    encode_output_path: str = field(
        default=None, metadata={"help": "encode output path"}
    )
    max_len: int = field(
        default=None, metadata={"help": "The maximum total input sequence length after tokenization. "
                                        "Use with caution, since it may truncate text prompts due to large image lengths."},
    )
    embedding_type: str = field(
        default="", metadata={"help": "embedding type"}
    )
    image_resolution: str = field(
        default='high', metadata={"help": "for models i.e. LLaVA-next and Qwen, resize images first"}
    )
    padding: bool = field(
        default=False,
    )
    additional_data: bool = field(
        default=False,
    )
    full_train_data: bool = field(
        default=False
    )
    llava_pretrain_data: int = field(
        default=0
    )
    
    
@dataclass
class TrainingArguments(TrainingArguments):
    image_encoder_freeze: bool = field(
        default=False, metadata={"help": "huggingface model name"}
    )
    output_dir: str = field(
        default=None, metadata={"help": "directory for saving trained models"}
    )
    project_name: str = field(
        default=None, metadata={"help": "project name"}
    )

    logging_steps: int = field(
        default=1, metadata={"help": "logging steps"}
    )
    num_train_epochs: int = field(
        default=1, metadata={"help": "number of training epochs"}
    )
    grad_cache: bool = field(
        default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(
        default=2, metadata={"help": "query side subset size"})
    gc_p_chunk_size: int = field(
        default=2, metadata={"help": "target side subset size"})

    use_vein: bool = field(
        default=False, 
    )
