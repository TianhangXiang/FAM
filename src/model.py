from typing import Dict, Optional
import torch
import logging
import torch.distributed as dist
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel
from src.arguments import ModelArguments, TrainingArguments
from src.model_utils import PHI3V, get_backbone_name, print_master, backbone2model
from src.utils import print_rank
from src.vlm_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
from src.vlm_backbone.llava_next import LlavaNextForConditionalGeneration
from src.vein_decoder import EmbeddingGuidedVisualReconstructor
import torch.nn.functional as F
from pdb import set_trace as b

class MMEBModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 model_args=None,
                 processor=None,
                 ):
        super().__init__()
        
        self.model_args = model_args
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        if model_args.temperature_learnable:
            self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        else:
            self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.loss_fn = self.get_normal_nce_loss
        
    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.encoder.gradient_checkpointing_enable({"use_reentrant": False})
        self.encoder.enable_input_require_grads()
    
    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)
    
    @classmethod
    def build(cls, model_args: ModelArguments, training_args: TrainingArguments=None, **kwargs):
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True, cache_dir=model_args.cache_dir)
        model_backbone = get_backbone_name(hf_config=config)
        setattr(model_args, 'model_backbone', model_backbone)
        print_master(f'Loading backbone [{model_backbone}]')
        # Loading the base model
        if model_backbone == PHI3V:
            config._attn_implementation = "eager"
            config.padding_side = "right"
            config.use_cache = False
            base_model = Phi3VForCausalLM.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                cache_dir=model_args.cache_dir,
            )
        elif model_backbone == LLAVA_NEXT:
            if model_args.not_use_flash_attn:
                config._attn_implementation = "sdpa"
                # config._attn_implementation = "eager"
            else:
                config._attn_implementation = "flash_attention_2"
            config.use_cache = False
            config.padding_side = "left"
            base_model = LlavaNextForConditionalGeneration.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                cache_dir=model_args.cache_dir,
            )
            
        else:
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name, **kwargs, config=config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                cache_dir=model_args.cache_dir,
                trust_remote_code=True)

        pretrain_dir = model_args.pretrain_dir
        if pretrain_dir is not None:
            lora_config = LoraConfig.from_pretrained(pretrain_dir)
            lora_model = PeftModel.from_pretrained(base_model, pretrain_dir, config=lora_config)
            base_model = lora_model.merge_and_unload()
            print_master(f"Loading pretrain ckpt from {pretrain_dir}")
                
        if model_args.lora:
            print_master(f'Loading lora adapter from {base_model}')
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules.split(','),
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False
            )
            lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                model_args=model_args,
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                model_args=model_args,
            )

        return model

    @classmethod
    def load(cls, model_args: ModelArguments, **kwargs):
        # Loading the base model
        checkpoint_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        setattr(model_args, 'model_backbone', model_backbone)
        print_master(f'Loading backbone [{model_backbone}]')

        if model_args.model_backbone in {LLAVA_NEXT}:
            config._attn_implementation = "flash_attention_2"
            config.vision_config._attn_implementation = "flash_attention_2"
            base_model = backbone2model[model_args.model_backbone].from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                config=config,
            )
        elif model_args.model_backbone == PHI3V:
            # Loading the base model
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True,)
            config.use_cache = False
            config.padding_side = "right"
            base_model = Phi3VForCausalLM.from_pretrained(model_args.model_name, **kwargs, config=config,
                                                          torch_dtype=torch.bfloat16, trust_remote_code=True)
            base_model.padding_side = "right"
        else:
            # Loading external base model from HF
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                checkpoint_path, **kwargs, config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,)
     
        pretrain_dir = model_args.pretrain_dir
        if pretrain_dir is not None:
            lora_config = LoraConfig.from_pretrained(pretrain_dir)
            lora_model = PeftModel.from_pretrained(base_model, pretrain_dir, config=lora_config)
            base_model = lora_model.merge_and_unload()
            print_master(f"Loading pretrain ckpt from {pretrain_dir}")
            
        # Building the model on top of the base
        if model_args.lora:
            lora_config = LoraConfig.from_pretrained(checkpoint_path)
            lora_model = PeftModel.from_pretrained(base_model, checkpoint_path, config=lora_config)

            merged_model = lora_model.merge_and_unload()
            model = cls(
                encoder=merged_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                model_args=model_args,
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                model_args=model_args,
            )

        return model
    
    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
    
    def get_normal_nce_loss(self, q_reps:torch.Tensor, t_reps:torch.Tensor):
        scores = self.compute_similarity(q_reps, t_reps)
        scores = scores.view(q_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (q_reps.size(0) // t_reps.size(0))
        loss = self.cross_entropy(scores / self.temperature, target)

        return loss
    
    def dense_encode_input(self, input):
        hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True, output_atentions=True)
        hidden_states = hidden_states.hidden_states
        pooled_output = self._pooling(hidden_states[-1], input['attention_mask'])
        
        return pooled_output, hidden_states
    
    def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, *args, **kwargs):
        qry_reps = self.encode_input(qry) if qry else None  # (bsz_per_device, dim)
        tgt_reps = self.encode_input(tgt) if tgt else None # (bsz_per_device, dim)
        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}

        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps
        
        loss_nce = self.loss_fn(all_qry_reps, all_tgt_reps)
        if self.is_ddp:
            loss = loss_nce * self.world_size
        
        output = {}
        output['loss'] = loss
        output['loss_nce'] = loss_nce.detach().clone()
        
        return output
    
    def encode_input(self, input):
        hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
        hidden_states = hidden_states.hidden_states[-1]
        pooled_output = self._pooling(hidden_states, input['attention_mask'])
        return pooled_output
    
    def encode_input_vis(self, input):
        output = self.encoder(**input, return_dict=True, output_hidden_states=True, output_attentions=True)
        hidden_states = output.hidden_states[-1]
        attention = output.attentions
        pooled_output = self._pooling(hidden_states, input['attention_mask'])
        return pooled_output, attention, output.hidden_states
    
    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling == 'last' or self.pooling == 'eos':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            batch_size = last_hidden_state.shape[0]
            if left_padding:
                # Get the vectors at the last position
                reps = last_hidden_state[torch.arange(batch_size), -1, :]
            else:
                # Calculate last 1 position in the original tensor
                eos_indices = attention_mask.sum(dim=1) - 1
                # Get the vectors at the last 1 position of each attention mask
                reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), eos_indices]
        else:
            raise NotImplementedError
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps


    def forward_for_vis(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, *args, **kwargs):
        qry_reps, qry_attention, qry_hidden_state = None, None, None
        tgt_reps, tgt_attention, tgt_hidden_state = None, None, None
        if qry:
            qry_reps, qry_attention, qry_hidden_state = self.encode_input_vis(qry) # (bsz_per_device, dim)
        if tgt:
            tgt_reps, tgt_attention, tgt_hidden_state = self.encode_input_vis(tgt) # (bsz_per_device, dim) 
            
        return {"qry_reps": qry_reps, "tgt_reps": tgt_reps, "qry_attention": qry_attention, "tgt_attention": tgt_attention, 'qry_hidden_state': qry_hidden_state, 'tgt_hidden_state': tgt_hidden_state}
    
class VEINMMEBModel(MMEBModel):
    def __init__(self, encoder: PreTrainedModel, pooling: str = 'cls', normalize: bool = False, temperature: float = 1.0, model_args=None):
        super().__init__(encoder, pooling, normalize, temperature, model_args)
        self.model_args = model_args
        self.visual_reconstructor = EmbeddingGuidedVisualReconstructor(
            input_dim=self.model_args.vein_mask_dim,
            mask_ratio=self.model_args.vein_mask_ratio,
            num_layers=self.model_args.vein_decoder_layers,
            reconstruct_strategy=self.model_args.vein_decoder_strategy,
            use_projector=self.model_args.vein_use_projector,
            mask_generate_type=self.model_args.vein_mask_generate_type,
            detach_feature=self.model_args.vein_detach_feature,
            add_pos_to_gt=self.model_args.vein_add_pos_to_target,
            dropout=self.model_args.vein_dropout,
            vein_norm_embedding=self.model_args.vein_norm_embedding,
            pos_type=self.model_args.vein_pos_type,
        ).to(torch.bfloat16)
        
        self.recons_loss_type = self.model_args.vein_loss_type
        self.to_recons_layer_id = self.model_args.vein_layer_to_apply
        self.recons_loss_weight = self.model_args.vein_loss_weight
    
    def forward_visual_reconstruciton(self, qry_reps, tgt_reps, qry_hidden_state, tgt_hidden_state, qry, tgt):
        # # gather all visual features from qry and tgt
        B = qry_reps.shape[0]
        
        qry_input_ids = qry['input_ids']
        tgt_input_ids = tgt['input_ids']
        
        viusal_features = []
        embedding_norm = []
        embedding_no_norm = []
        
        model_name = self.model_args.model_name
        
        if 'Phi-3.5' in model_name:
            for i in range(B):
                image = qry['images'][i]
                need_recons = qry['pure_vision'][i]
                
                if image and need_recons:
                    input_ids = qry['input_ids'][i]
                    image_token_index = (input_ids == -1)
                    
                    viusal_features.append(qry_hidden_state[self.to_recons_layer_id][i][image_token_index].unsqueeze(0))
                    embedding_norm.append(qry_reps[i].unsqueeze(0))
                    embedding_no_norm.append(qry_reps[i].unsqueeze(0))
                    
            for i in range(B):
                image = tgt['images'][i]
                need_recons = tgt['pure_vision'][i]
                if image and need_recons:
                    input_ids = tgt['input_ids'][i]
                    image_token_index = (input_ids == -1)
                    
                    viusal_features.append(tgt_hidden_state[self.to_recons_layer_id][i][image_token_index].unsqueeze(0))
                    embedding_norm.append(tgt_reps[i].unsqueeze(0))
                    embedding_no_norm.append(tgt_reps[i].unsqueeze(0))
        else:
            # only support left padding
            if not self.model_args.vein_on_only_pure_vision_task:
                for i in range(B):
                    has_image = (151655 in qry_input_ids[i]) and (qry['images'][i] is not None)
                    if has_image:
                        image_token_index = (qry_input_ids[i] == 151655)
                        viusal_features.append(qry_hidden_state[self.to_recons_layer_id][i][image_token_index].unsqueeze(0))
                        embedding_no_norm.append(qry_hidden_state[self.to_recons_layer_id][i][-1].unsqueeze(0))
                        embedding_norm.append(qry_reps[i].unsqueeze(0))
                        
                for i in range(B):
                    has_image = 151655 in tgt_input_ids[i] and (tgt['images'][i] is not None)
                    if has_image:
                        image_token_index = (tgt_input_ids[i] == 151655)
                        viusal_features.append(tgt_hidden_state[self.to_recons_layer_id][i][image_token_index].unsqueeze(0))
                        embedding_no_norm.append(tgt_hidden_state[self.to_recons_layer_id][i][-1].unsqueeze(0))
                        embedding_norm.append(tgt_reps[i].unsqueeze(0))
            else:
                ### process query
                for i in range(B):
                    has_image = (151655 in qry_input_ids[i]) and (qry['images'][i] is not None)
                    need_recons = qry['pure_vision'][i]
                    if has_image and need_recons:
                        image_token_index = (qry_input_ids[i] == 151655)
                        viusal_features.append(qry_hidden_state[self.to_recons_layer_id][i][image_token_index].unsqueeze(0))
                        embedding_no_norm.append(qry_hidden_state[self.to_recons_layer_id][i][-1].unsqueeze(0))
                        embedding_norm.append(qry_reps[i].unsqueeze(0))
                        
                ### process tgt
                for i in range(B):
                    has_image = 151655 in tgt_input_ids[i] and (tgt['images'][i] is not None)
                    need_recons = tgt['pure_vision'][i]
                    if has_image and need_recons:
                        image_token_index = (tgt_input_ids[i] == 151655)
                        viusal_features.append(tgt_hidden_state[self.to_recons_layer_id][i][image_token_index].unsqueeze(0))
                        embedding_no_norm.append(tgt_hidden_state[self.to_recons_layer_id][i][-1].unsqueeze(0))
                        embedding_norm.append(tgt_reps[i].unsqueeze(0))
                        
        if len(viusal_features) != 0:  
            viusal_features = torch.cat(viusal_features, dim=0)
            embedding_no_norm = torch.cat(embedding_no_norm, dim=0)
            embedding_norm = torch.cat(embedding_norm, dim=0)
        else:
            viusal_features = None
            embedding_no_norm = None
            embedding_norm = None
        
        reconstructed_visual_feats, targets_feats = self.visual_reconstructor(viusal_features, embedding_no_norm, embedding_norm)
        
        return reconstructed_visual_feats, targets_feats
    
    def compute_visual_reconstruction_loss(self, reconstructed_visual_feats, targets_feats):
        if self.recons_loss_type == 'cos':
            if reconstructed_visual_feats is not None:
                cos_sim = F.cosine_similarity(reconstructed_visual_feats, targets_feats, dim=-1).mean()
                loss = 1 - cos_sim
            else:
                loss = None
                
        elif self.recons_loss_type == 'mse':
            if reconstructed_visual_feats is not None:
                loss = F.mse_loss(reconstructed_visual_feats, targets_feats)
            else:
                loss = None
            
        return loss
    

    # def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, *args, **kwargs):
        
    #     qry_reps, qry_hidden_state = self.dense_encode_input(qry) if qry else None
    #     tgt_reps, tgt_hidden_state = self.dense_encode_input(tgt) if tgt else None
        
    #     reconstructed_visual_feats, targets_feats = self.forward_visual_reconstruciton(qry_reps, tgt_reps, qry_hidden_state, tgt_hidden_state, qry, tgt)
    #     loss_recons = self.compute_visual_reconstruction_loss(reconstructed_visual_feats, targets_feats) * self.recons_loss_weight 

    #     if self.is_ddp:
    #         all_qry_reps = self._dist_gather_tensor(qry_reps)
    #         all_tgt_reps = self._dist_gather_tensor(tgt_reps)
    #     else:
    #         all_qry_reps = qry_reps
    #         all_tgt_reps = tgt_reps
        
    #     loss_nce = self.loss_fn(all_qry_reps, all_tgt_reps)
        
    #     if loss_recons is not None:
    #         loss = loss_nce + loss_recons
        
    #     if self.is_ddp:
    #         loss = loss * self.world_size
            
    #     output = {
    #         'loss': loss,
    #         'loss_nce': loss_nce.detach().clone(),
    #         'loss_recons': loss_recons.detach().clone() if loss_recons is not None else None,
    #     }
        
    #     return output

    def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, *args, **kwargs):
        
        qry_reps, qry_hidden_state = self.dense_encode_input(qry) if qry else None
        tgt_reps, tgt_hidden_state = self.dense_encode_input(tgt) if tgt else None
        
        reconstructed_visual_feats, targets_feats = self.forward_visual_reconstruciton(qry_reps, tgt_reps, qry_hidden_state, tgt_hidden_state, qry, tgt)
        
        loss_recons = self.compute_visual_reconstruction_loss(reconstructed_visual_feats, targets_feats)
        
        if loss_recons is not None:
            loss_recons = loss_recons * self.recons_loss_weight 
            
        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps

        loss_nce = self.loss_fn(all_qry_reps, all_tgt_reps)
        
        if loss_recons is not None:
            loss = loss_nce + loss_recons
        else:
            loss = loss_nce
            
        if self.is_ddp:
            loss = loss * self.world_size
        
        output = {
            'loss': loss,
            'loss_nce': loss_nce.detach().clone(),
            'loss_recons': loss_recons.detach().clone() if loss_recons is not None else None,
        }
        
        return output

class PretrainMMEBModel(MMEBModel):
    def __init__(self, encoder: PreTrainedModel, pooling: str = 'cls', normalize: bool = False, temperature: float = 1.0, model_args=None):
        super().__init__(encoder, pooling, normalize, temperature, model_args)
        # TODO: config the temperature
        self.model_args = model_args
        
        # self.processor = model_args
        # self.v_temperature = nn.Parameter(torch.tensor(14.28))
        # self.t_temperature = nn.Parameter(torch.tensor(14.28))

    def forward_vision_contrastive_loss(self, embedding, vision_feats):
        """
        text embedding: [B, D]
        vision_feats: [B, L, D]
        """
        B, L, D = vision_feats.shape
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        vision_feats = torch.nn.functional.normalize(vision_feats, p=2, dim=-1)
        # (B, D) * (D, B*L) ==> (B, B*L) ==> (B, B, L)
        cos_sim = torch.matmul(embedding, vision_feats.view(-1, D).T).view(B, B, -1)
        # (B, B, L) ==> (B, B)
        scores = cos_sim.mean(dim=-1)
        scores = scores / self.temperature
        target = torch.arange(B, device=embedding.device)

        loss = self.cross_entropy(scores, target)
        
        return loss, scores

    def forward_text_contrastive_loss(self, embedding, text_feats, attention_mask):
        """
        embedding: [B, D]
        text_feats: [B, L, D]
        attention_mask: [B, D]
        """
        B, L, D = text_feats.shape
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        text_feats = torch.nn.functional.normalize(text_feats, p=2, dim=-1)
        text_feats = text_feats * attention_mask.unsqueeze(-1)
        
        # (B, D) * (D, B*L) ==> (B, B*L) ==> (B, B, L)
        cos_sim = torch.matmul(embedding, text_feats.view(-1, D).T).view(B, B, -1)
        
        # (B, B, L) ==> (B, B)
        scores = cos_sim.sum(dim=-1) / attention_mask.sum(dim=-1)
        
        scores = scores / self.temperature
        target = torch.arange(B, device=embedding.device)

        loss = self.cross_entropy(scores, target)
        
        return loss, scores
    
    def forward_cross_contrastive_loss(self, vision_feats, text_feats, attention_mask):
        """
        vision_feats: [B, N, D]
        text_feats: [B, L, D]
        attention_mask: [B, L]  (1 for valid tokens, 0 for padding)
        """
        B, N, D = vision_feats.shape
        _, L, _ = text_feats.shape

        # 归一化
        vision_feats = F.normalize(vision_feats, p=2, dim=-1)  # [B, N, D]
        text_feats = F.normalize(text_feats, p=2, dim=-1)      # [B, L, D]

        # 利用attention_mask屏蔽无效文本token
        text_feats = text_feats * attention_mask.unsqueeze(-1)  # [B, L, D]

        # (B, N, D) ==> (B*N, D)
        vision_feats_c = vision_feats.view(B*N, D)
        # (B, L, D) ==> (B*L, D)
        text_feats_c = text_feats.view(B*L, D)
        
        # (B*N, B*L)
        sim_matrix = torch.matmul(vision_feats_c, text_feats_c.T)
        
        # (B*N, B*L) ==> (B*N, B, L) ==> (B, N, B, L) ==> (B, B, N, L)
        sim_matrix = sim_matrix.contiguous().view(B*N, B, L).view(B, N, B, L).transpose(1, 2)
        
        # mean in text feats
        # sim_matrix: (B, B, N, L) ==> (B, B, N) 
        # attention_mask: (B, L) ==> (B, ) ==> (B, 1) ==> (B, B) ==> (B, B, 1)
        sim_matrix_mean = sim_matrix.sum(dim=-1) / attention_mask.sum(dim=-1).unsqueeze(0).expand(B, B).unsqueeze(2)
        
        # sim_debug = sim_matrix.mean(dim=-1)
        # mean in vision feats
        # (B, B, N) ==> (B, B)
        sim_matrix_mean = sim_matrix_mean.mean(dim=-1)

        scores = sim_matrix_mean / self.temperature
        target = torch.arange(B, device=vision_feats.device)

        loss = self.cross_entropy(scores, target)
        
        return loss, scores
    
    def forward_filip_contrastive_loss(self, vision_feats, text_feats, attention_mask):
        """
        vision_feats: [B, N, D]
        text_feats: [B, L, D]
        attention_mask: [B, L]  (1 for valid tokens, 0 for padding)
        """
        B, N, D = vision_feats.shape
        _, L, _ = text_feats.shape

        # 归一化
        vision_feats = F.normalize(vision_feats, p=2, dim=-1)  # [B, N, D]
        text_feats = F.normalize(text_feats, p=2, dim=-1)      # [B, L, D]

        # 利用attention_mask屏蔽无效文本token
        text_feats = text_feats * attention_mask.unsqueeze(-1)  # [B, L, D]

        # (B, N, D) ==> (B*N, D)
        vision_feats_c = vision_feats.view(B*N, D)
        # (B, L, D) ==> (B*L, D)
        text_feats_c = text_feats.view(B*L, D)
        
        # (B*N, B*L)
        sim_matrix = torch.matmul(vision_feats_c, text_feats_c.T)
        
        # (B*N, B*L) ==> (B*N, B, L) ==> (B, N, B, L) ==> (B, B, N, L)
        sim_matrix = sim_matrix.contiguous().view(B*N, B, L).view(B, N, B, L).transpose(1, 2)
        
        ######## max in text feats and mean in vision and get scores
        sim_matrix_max_text_mean_vision = sim_matrix.max(dim=-1).values.mean(dim=-1)
        
        ####### max in vision and mean in text feats and get scores
        sim_martix_max_vision_mean_text = sim_matrix.max(dim=2).values
        sim_martix_max_vision_mean_text = sim_martix_max_vision_mean_text.sum(dim=-1) / attention_mask.sum(dim=-1).unsqueeze(0).expand(B, B)
        
        sim_matrix_max_text_mean_vision = sim_matrix_max_text_mean_vision / self.temperature
        sim_martix_max_vision_mean_text = sim_martix_max_vision_mean_text / self.temperature
        
        target = torch.arange(B, device=vision_feats.device)

        loss_1 = self.cross_entropy(sim_matrix_max_text_mean_vision, target)
        loss_2 = self.cross_entropy(sim_martix_max_vision_mean_text, target)
        loss = (loss_1 + loss_2) / 2
        return loss, None


    def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, *args, **kwargs):
        
        qry_reps, qry_hidden_state = self.dense_encode_input(qry) if qry else None
        tgt_reps, tgt_hidden_state = self.dense_encode_input(tgt) if tgt else None
        model_name = self.model_args.model_name        
        # TODO: config the layer id
        qry_hidden_state = qry_hidden_state[-1]
        tgt_hidden_state = tgt_hidden_state[-1]

        B = qry_reps.shape[0]
                  
        if 'Phi-3.5' in model_name:
            
            # for Phi-3.5, ':' token id is 29901
            start_token = 25 # :
            
            ###################### prepare vision feats #########################
            vision_feats = []
            qry_input_ids = qry['input_ids']
            for i in range(len(qry_input_ids)):
                img_feats_index = qry_input_ids[i] == -1
                vision_feats.append(qry_hidden_state[i][img_feats_index].unsqueeze(0))
            vision_feats = torch.cat(vision_feats, dim=0)
            
            ###################### prepare text feats and attention mask #########################
            tgt_input_ids = tgt['input_ids']
            text_feats_list = []
            
            # for Phi-3.5, ':' token id is 29901
            start_token = 29901 # :
            for i in range(len(tgt_input_ids)):
                input_ids = tgt_input_ids[i]
                if start_token not in input_ids:
                    import pdb; pdb.set_trace()
                    print(self.processor.tokenizer.decode(input_ids))
                
                start_index = (input_ids == start_token).nonzero(as_tuple=True)[0]
                end_index = tgt['attention_mask'][i].nonzero(as_tuple=True)[0][-1]
                text_tokens = tgt_hidden_state[i][start_index + 1: end_index]
                text_feats_list.append(text_tokens)
                
        elif 'llava' in model_name:
            pass
        else:
            pass
        
        # NOTE: 由于每个 text 的长度不一样，需要先paddding 成一个完整的 tensor，然后用 paddding mask来表征padding的位置，1 表示正常，2表示 padding
        max_length = 64
        _, D = text_feats_list[0].shape
        device = text_feats_list[0].device
        
        padded_text_feats = torch.zeros(B, max_length, D, device=device)
        padding_mask = torch.zeros(B, max_length, device=device).bool()
        
        for b_id, text_feat in enumerate(text_feats_list):
            L, D = text_feat.shape
            if L <= max_length:
                padded_text_feats[b_id][:L] = text_feat
                padding_mask[b_id][:L] = 1
                assert padded_text_feats[b_id][padding_mask[b_id]].sum() == text_feat.sum()
            else:
                padded_text_feats[b_id] = text_feat[:max_length]
                padding_mask[b_id] = 1
                assert padded_text_feats[b_id][padding_mask[b_id]].sum() == text_feat[:max_length].sum()
            
        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
            all_vision_feats = self._dist_gather_tensor(vision_feats)
            all_padding_text_feats = self._dist_gather_tensor(padded_text_feats)
            all_padding_masks = self._dist_gather_tensor(padding_mask)
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps
            all_vision_feats = vision_feats
            all_padding_text_feats = padded_text_feats
            all_padding_masks = padding_mask
        
        # print_rank(f"visial feat shape: {vision_feats.shape}")
        # print_rank(f"padded_text_feats shape: {padded_text_feats.shape}")
        # print_rank(f"padding_mask shape: {padding_mask.shape}")
        # print_rank(f"gather visial feat shape: {all_vision_feats.shape}")
        # print_rank(f"gather padded_text_feats shape: {all_padding_text_feats.shape}")
        # print_rank(f"gather padding_mask shape: {all_padding_masks.shape}")

        loss_v_cons, v_scores = self.forward_vision_contrastive_loss(all_tgt_reps, all_vision_feats)
        # import pdb; pdb.set_trace()
        
        loss_t_cons, t_scores = self.forward_text_contrastive_loss(all_qry_reps, all_padding_text_feats, all_padding_masks)
        # import pdb; pdb.set_trace()
        
        locc_c_cons, c_scores = self.forward_cross_contrastive_loss(all_vision_feats, all_padding_text_feats, all_padding_masks)
        # locc_c_cons, c_scores = self.forward_filip_contrastive_loss(all_vision_feats, all_padding_text_feats, all_padding_masks)
        
        # print_master(f"loss_v_cons: {loss_v_cons}")
        # print_master(f"loss_t_cons: {loss_t_cons}")
        
        output = {}
        
        loss_nce = self.loss_fn(all_qry_reps, all_tgt_reps)
        output['loss_nce'] = loss_nce.detach().clone()
        
        loss = loss_nce
        
        if self.model_args.pretrain_use_vision_expand_loss:
            loss += loss_v_cons
            output['loss_v_cons'] = loss_v_cons.detach().clone()
            
        if self.model_args.pretrain_use_text_expand_loss:
            loss += loss_t_cons
            output['loss_t_cons'] = loss_t_cons.detach().clone()
        
        if self.model_args.pretrain_use_cross_loss:
            loss += locc_c_cons
            output['loss_c_cons'] = locc_c_cons.detach().clone()
        
        # locc_filip_cons, _ = self.forward_filip_contrastive_loss(all_vision_feats, all_padding_text_feats, all_padding_masks)
        # if self.model_args.pretrain_use_filip_loss:
        #     loss += locc_filip_cons
        #     output['loss_filip_cons'] = locc_c_cons.detach().clone()
            
        if self.is_ddp:
            loss = loss * self.world_size
        
        output['loss'] = loss
        
        return output