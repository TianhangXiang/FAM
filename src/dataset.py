import re
import sys
import copy
import pandas as pd
import json
from typing import List, Tuple

import datasets
from datasets import load_dataset, concatenate_datasets
from PIL import Image
import os

from torch.jit import isinstance

from src.model_utils import PHI3V, vlm_image_tokens
from src.utils import print_master, print_rank

# from datasets import Dataset  # @ruimeng, still buggy
from torch.utils.data import Dataset
from collections import OrderedDict
import random

def process_image(image, resolution, max_dim=1344):
    if image is None:
        return None
    if resolution == "high":
        image = image.resize((1344, 1344))
    elif resolution == "mid":
        image = image.resize((672, 672))
    elif resolution == "low":
        image = image.resize((336, 336))
    else:
        cur_max_dim = max(image.size)
        if cur_max_dim > max_dim:
            image = image.resize((max_dim, max_dim))
    return image

class TrainJsonDataset(Dataset):
    def __init__(self, data_args, model_args):
        self.data_args = data_args
        self.model_args = model_args
        self.train_data = []
        print_rank(f"Loading {len(data_args.subset_name)} datasets: {data_args.subset_name}")
        self.qry_dataset_pure_vision_mapping = {
            'ImageNet_1K': True,
            'HatefulMemes': False,
            'N24News': True,
            'VOC2007': True,
            'SUN397': True,
            
            'OK-VQA': False,
            'A-OKVQA': False,
            'DocVQA': False,
            'InfographicsVQA': False,
            'ChartQA': False,
            'Visual7W': False,
            
            'VisDial': False,
            'CIRR': False,
            'VisualNews_t2i': False,
            'VisualNews_i2t': True,
            'MSCOCO_t2i': False,
            'MSCOCO_i2t': True,
            'NIGHTS': True,
            'WebQA': False,
            
            'MSCOCO': False,
        }

        self.tgt_dataset_pure_vision_mapping = {
            'ImageNet_1K': False,
            'HatefulMemes': False,
            'N24News': False,
            'VOC2007': False,
            'SUN397': False,
            
            'OK-VQA': False,
            'A-OKVQA': False,
            'DocVQA': False,
            'InfographicsVQA': False,
            'ChartQA': False,
            'Visual7W': False,
            
            'VisDial': True,
            'CIRR': True,
            'VisualNews_t2i': True,
            'VisualNews_i2t': False,
            'MSCOCO_t2i': True,
            'MSCOCO_i2t': False,
            'NIGHTS': True,
            'WebQA': False,
            
            'MSCOCO': True,
        } 
        
        # loading SFT data
        for subset in data_args.subset_name:
            jsonl_file_path = os.path.join('./CoTEmbedding/data', f'{subset}.jsonl')
            data = []
            with open(jsonl_file_path, 'r') as file:
                for line in file:
                    data.append(json.loads(line))
                if not self.data_args.full_train_data:
                    print_rank(f"Loading {len(data[:50000])} items from {subset}")
                    self.train_data.extend(data[:50000])   
                else:
                    print_rank(f"Loading {len(data[:100000])} items from {subset}")
                    self.train_data.extend(data[:100000])
    
        self.llava_data = []
        if data_args.llava_pretrain_data != 0:
            json_path = "./EmbeddingKit/data/GCC/metadata.json"
                    
            with open(json_path, 'r') as file:
                self.llava_data = json.load(file)[:data_args.llava_pretrain_data]

            print_rank(f"Loading {len(self.llava_data)} items from llava pretrain data")
            
            self.image_instruction_pool = [
                "Represent the given image",
                "Represent the given image for classification",
                "Indentify the object in the image",
                "Indentify the scene shown in the image",
                "Represent what you can see in the image",
            ]
            
            self.text_instruction_pool = [
                "Represent the given text into a vector",
                "Represent the given sentence",
                "Represent the given text label",
                "Indentify the given sentence",
                "Imagine the scene that the setence describe",
                "Imagine the scene that the text describe",
            ]

        print_rank(f"Totally Loaded {len(self)} items")
        
    def __len__(self):
        return len(self.train_data) + len(self.llava_data)

    def _get_image(self, img_path):
        if not img_path:
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        backbone = self.model_args.model_backbone
        if backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image
    
    def _get_llava_image(self, img_path):
        if img_path == "":
            return None
        image_dir = './EmbeddingKit/data/GCC/GCC_image'
        full_img_path = os.path.join(image_dir, img_path)
        image = Image.open(full_img_path)
        if self.model_args.model_backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image

    def __getitem__(self, data_idx) -> Tuple[str, List[str]]:
        # load SFT data
        if data_idx < len(self.train_data):
            qry_texts, qry_image_paths, pos_texts, pos_image_paths = (
                self.train_data[data_idx]["qry"], self.train_data[data_idx]["qry_image_path"],
                self.train_data[data_idx]["pos_text"], self.train_data[data_idx]["pos_image_path"]
            )
            neg_texts, neg_image_paths = '', None
            
            if isinstance(data_idx, int):
                qry_texts = [qry_texts]
                qry_image_paths = [qry_image_paths]
                pos_texts = [pos_texts]
                pos_image_paths = [pos_image_paths]
                neg_texts = [neg_texts]
                neg_image_paths = [neg_image_paths]
            _qry_texts, _qry_images, _pos_texts, _pos_images, _neg_texts, _neg_images = [], [], [], [], [], []
            backbone = self.model_args.model_backbone
            for qry_text, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path \
                in zip(qry_texts, qry_image_paths, pos_texts, pos_image_paths, neg_texts, neg_image_paths):
                # instructions were hardcoded with Phi3 image special tokens
                # Update image token for llava and colqwen2
                if backbone != PHI3V:
                    qry_text = qry_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone])
                    pos_text = pos_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone])
                    neg_text = neg_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone]) if neg_text else None
                qry_image = self._get_image(qry_image_path)
                pos_image = self._get_image(pos_image_path)
                neg_image = self._get_image(neg_image_path) if neg_image_path else None
                if (not qry_text and not qry_image) or (not pos_text and not pos_image):
                    print("empty inputs")
                    continue
                _qry_texts.append(qry_text)
                _qry_images.append(qry_image)
                _pos_texts.append(pos_text)
                _pos_images.append(pos_image)
                _neg_texts.append(neg_text)
                _neg_images.append(neg_image)
            
            dataset = self.train_data[data_idx]['dataset']
            qry_pure_vision_indicator = self.qry_dataset_pure_vision_mapping[dataset]
            tgt_pure_vision_indicator = self.tgt_dataset_pure_vision_mapping[dataset]
            
            return {"query_text": _qry_texts, "query_image": _qry_images,
                    "pos_text": _pos_texts, "pos_image": _pos_images,
                    "neg_text": _neg_texts, "neg_image": _neg_images,
                    "qry_pure_vision_indicator": qry_pure_vision_indicator, "tgt_pure_vision_indicator": tgt_pure_vision_indicator,
                    "dataset": dataset,}
            
        # load llava data
        else:
            data_idx = data_idx - len(self.train_data)
            caption = self.llava_data[data_idx]['blip_caption']
            # caption = self.llava_data[data_idx].get('dense_caption', None)
            if caption is not None:
                caption = caption[0]
            else:
                caption = self.llava_data[data_idx]['caption']
                
            img_path = self.llava_data[data_idx]['image']
            
            image_text_input = "<|image_1|>" + random.choice(self.image_instruction_pool) + " \n"
            text_input = random.choice(self.text_instruction_pool) + ": " + caption + " \n"
            
            backbone = self.model_args.model_backbone
            
            if backbone != PHI3V:
                image_text_input = image_text_input.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone])
        
            return {"query_text": image_text_input, "query_image":  self._get_llava_image(img_path),
                "pos_text": text_input, "pos_image": None,
                "neg_text": None, "neg_image": None,
                "qry_pure_vision_indicator": True, "tgt_pure_vision_indicator": False,
                "dataset": "llava",}


class EvalDataset(Dataset):
    def __init__(self, data_args, model_args, subset, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        self.data_args = data_args
        self.model_args = model_args
        self.backbone = self.model_args.model_backbone

        self.eval_data = load_dataset(
            self.data_args.dataset_name,
            subset,
            split=self.data_args.dataset_split,
        )
        self.paired_data = self.get_paired_data(text_field, img_path_field)
        self.paired_dataset = datasets.Dataset.from_dict({
            "text": [pair["text"] for pair in self.paired_data],
            "img_path": [pair["img_path"] for pair in self.paired_data]
        })

    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, item):
        text, img_path = self.paired_dataset[item]["text"], self.paired_dataset[item]["img_path"]
        # text = "<|image_1|> Describe the image in details \n Assitance: "
        if self.backbone != PHI3V:
            text = text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.backbone])

        return text, self._get_image(img_path),

    def _get_image(self, img_path):
        if img_path == "":
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        
        return process_image(image, self.data_args.image_resolution)
    
        # if self.model_args.model_backbone != PHI3V and self.data_args.image_resolution:
        #     return process_image(image, self.data_args.image_resolution)
        # else:
        #     return image
        # return image

    def get_paired_data(self, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        unique_pair = OrderedDict()  # Use OrderedDict to preserve insertion order
        for row in self.eval_data:
            if isinstance(row[text_field], str):
                if row[text_field]:
                    unique_pair[(row[text_field], row[img_path_field])] = None # Use a dummy value, only keys matter
                else:
                    if isinstance(row[img_path_field], List):
                        for img_path in row[img_path_field]:
                            unique_pair[(row[text_field], img_path)] = None
                    else:
                        unique_pair[(row[text_field], row[img_path_field])] = None
            elif type(row[text_field]) == list:
                assert type(row[img_path_field]) == list and len(row[img_path_field]) == len(row[text_field])
                for text, img_path in zip(row[text_field], row[img_path_field]):
                    unique_pair[(text, img_path)] = None

        paired_data = [{"text": text, "img_path": img_path} for (text, img_path) in unique_pair.keys()]
        return paired_data

class PretrainDataset(Dataset):
    """
    this dataset is used for scale up the whole data
    """
    def __init__(self, data_args, model_args):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        self.data_args = data_args
        self.model_args = model_args
        self.backbone = self.model_args.model_backbone

        gcc_json_path = "./EmbeddingKit/data/GCC/metadata.json"
        with open(gcc_json_path, 'r') as file:
            self.gcc_data = json.load(file)
        
        self.gcc_image_instruction_pool = [
            "Represent the given image",
            "Represent the given image into one word",
            "Represent the given image for classification",
            "Indentify the object in the image",
            "Indentify the scene shown in the image",
            "Represent what you can see in the image",
            "Summary the image into one word",
        ]
        
        self.gcc_text_instruction_pool = [
            "Represent the given text into a vector",
            "Represent the given sentence",
            "Represent the given sentence into one word",
            "Represent the given words",
            "Indentify the given sentence",
            "Imagine the scene that the setence describe",
            "Imagine the scene that the text describe",
            "Summary the text into one word",
        ]

        self.ft_data = []
        
    def __len__(self):
        return len(self.gcc_data) + len(self.ft_data)

    def __getitem__(self, data_idx):
        # return pretrain data
        if data_idx < len(self.gcc_data):
            caption = self.gcc_data[data_idx]['blip_caption']
            if caption is None:
                caption = self.gcc_data[data_idx]['caption']
                
            img_path = self.gcc_data[data_idx]['image']
            
            image_text_input = "<|image_1|>" + random.choice(self.gcc_image_instruction_pool) + "\n"
            text_input = random.choice(self.gcc_text_instruction_pool) + ": " + caption + "\n"
            
            if self.backbone != PHI3V:
                image_text_input = image_text_input.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.backbone])
        
            return {"query_text": image_text_input, "query_image":  self._get_pretrain_image(img_path),
                "pos_text": text_input, "pos_image": None,
                "neg_text": None, "neg_image": None,
                "dataset": "llava", 
                "qry_pure_vision_indicator": True, "tgt_pure_vision_indicator": False,}
            
        # return fine tuning data
        else:
            data_idx = data_idx - len(self.gcc_data)
            qry_texts, qry_image_paths, pos_texts, pos_image_paths = (
                self.ft_data[data_idx]["qry"], self.ft_data[data_idx]["qry_image_path"],
                self.ft_data[data_idx]["pos_text"], self.ft_data[data_idx]["pos_image_path"]
            )
            if 'neg_text' in self.ft_data.column_names:
                neg_texts, neg_image_paths = self.ft_data[data_idx]["neg_text"], self.ft_data[data_idx]["neg_image_path"]
            else:
                neg_texts, neg_image_paths = [''] * len(data_idx), [] * len(data_idx)
            if isinstance(data_idx, int):
                qry_texts = [qry_texts]
                qry_image_paths = [qry_image_paths]
                pos_texts = [pos_texts]
                pos_image_paths = [pos_image_paths]
                neg_texts = [neg_texts]
                neg_image_paths = [neg_image_paths]
            _qry_texts, _qry_images, _pos_texts, _pos_images, _neg_texts, _neg_images = [], [], [], [], [], []
            backbone = self.model_args.model_backbone
            for qry_text, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path \
                in zip(qry_texts, qry_image_paths, pos_texts, pos_image_paths, neg_texts, neg_image_paths):
                # instructions were hardcoded with Phi3 image special tokens
                # Update image token for llava and colqwen2
                if backbone != PHI3V:
                    qry_text = qry_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone])
                    pos_text = pos_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone])
                    neg_text = neg_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone]) if neg_text else None
                qry_image = self._get_image(qry_image_path)
                pos_image = self._get_image(pos_image_path)
                neg_image = self._get_image(neg_image_path) if neg_image_path else None
                if (not qry_text and not qry_image) or (not pos_text and not pos_image):
                    print("empty inputs")
                    continue
                _qry_texts.append(qry_text)
                _qry_images.append(qry_image)
                _pos_texts.append(pos_text)
                _pos_images.append(pos_image)
                _neg_texts.append(neg_text)
                _neg_images.append(neg_image)

            return {"query_text": _qry_texts, "query_image": _qry_images,
                    "pos_text": _pos_texts, "pos_image": _pos_images,
                    "neg_text": _neg_texts, "neg_image": _neg_images}

    def _get_pretrain_image(self, img_path):
        if img_path == "":
            return None
        image_dir = './EmbeddingKit/data/GCC/GCC_image'
        full_img_path = os.path.join(image_dir, img_path)
        image = Image.open(full_img_path)
        
        return process_image(image, self.data_args.image_resolution)
        # if self.model_args.model_backbone != PHI3V and self.data_args.image_resolution:
        #     return process_image(image, self.data_args.image_resolution)
        # else:
        #     return image
        
    def _get_image(self, img_path):
        if img_path == "":
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        return process_image(image, self.data_args.image_resolution)
    
        # if self.model_args.model_backbone != PHI3V and self.data_args.image_resolution:
        #     return process_image(image, self.data_args.image_resolution)
        # else:
        #     return image

if __name__ == "__main__":
   pass


