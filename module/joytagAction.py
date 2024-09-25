import huggingface_hub
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, \
    AutoModelForCausalLM
from pathlib import Path
import torch
from torch.nn.parallel import DataParallel
import torch.amp.autocast_mode
from PIL import Image
from waifuc.model import ImageItem
from waifuc.action import ProcessAction,FilterAction
from waifuc.source.base import BaseDataSource
import os
torch.set_default_device("cuda")
CLIP_PATH = "google/siglip-so400m-patch14-384"
VLM_PROMPT = "A descriptive caption for this image:\n"
MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"
project_root = os.path.dirname(os.path.abspath(__file__)).split("module")[0]
CHECKPOINT_PATH = os.path.join(project_root,'model',"joytag")

HF_TOKEN = os.environ.get("HF_TOKEN", None)
class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)

    def forward(self, vision_outputs: torch.Tensor):
        x = self.linear1(vision_outputs)
        x = self.activation(x)
        x = self.linear2(x)
        return x
class JoytagAction(ProcessAction):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        torch.set_default_device(self.device)
        
        self.CLIP_PATH = "google/siglip-so400m-patch14-384"
        self.VLM_PROMPT = "A descriptive caption for this image:\n"
        self.MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.project_root = os.path.dirname(os.path.abspath(__file__)).split("module")[0]
        self.CHECKPOINT_PATH = os.path.join(self.project_root, 'model', "joytag")

        self._load_models()

    def _load_models(self):
        print("加载 CLIP")
        self.clip_processor = AutoProcessor.from_pretrained(self.CLIP_PATH)
        self.clip_model = AutoModel.from_pretrained(self.CLIP_PATH).vision_model
        self.clip_model.eval().requires_grad_(False).to(self.device)

        huggingface_hub.login(token="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")

        print("加载分词器")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH, use_fast=False)

        print("加载 LLM")
        self.text_model = AutoModelForCausalLM.from_pretrained(self.MODEL_PATH, torch_dtype=torch.bfloat16)
        self.text_model.eval().to(self.device)

        print("加载图像适配器")
        self.image_adapter = ImageAdapter(self.clip_model.config.hidden_size, self.text_model.config.hidden_size)
        self.image_adapter.load_state_dict(torch.load(os.path.join(self.CHECKPOINT_PATH, "image_adapter.pt"), map_location=self.device))
        self.image_adapter.eval().to(self.device)
        self.clip_model = DataParallel(self.clip_model)
        self.image_adapter = DataParallel(self.image_adapter)
        self.text_model = DataParallel(self.text_model)

    @torch.no_grad()
    def _generate_caption(self, image):
        torch.cuda.empty_cache()
        
        # 预处理图片
        images = self.clip_processor(images=image, return_tensors='pt', padding=True).pixel_values
        images = images.to(self.device)

        # 标记提示词
        prompt = self.tokenizer.encode(self.VLM_PROMPT, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False)
        prompt = prompt.to(self.device)

        # 嵌入图片
        with torch.amp.autocast_mode.autocast(self.device, enabled=True):
            vision_outputs = self.clip_model(pixel_values=images, output_hidden_states=True)
            image_features = vision_outputs.hidden_states[-2]
            embedded_images = self.image_adapter(image_features)
            embedded_images = embedded_images.to(self.device)

        # 嵌入提示词
        prompt_embeds = self.text_model.module.model.embed_tokens(prompt)
        embedded_bos = self.text_model.module.model.embed_tokens(
            torch.tensor([[self.tokenizer.bos_token_id]], device=self.device, dtype=torch.int64))

        # 构建提示
        batch_size = embedded_images.shape[0]
        inputs_embeds = torch.cat([
            embedded_bos.expand(batch_size, -1, -1),
            embedded_images.to(dtype=embedded_bos.dtype),
            prompt_embeds.expand(batch_size, -1, -1),
        ], dim=1)

        input_ids = torch.cat([
            torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long).expand(batch_size, -1),
            torch.zeros((batch_size, embedded_images.shape[1]), dtype=torch.long),
            prompt.expand(batch_size, -1),
        ], dim=1).to(self.device)
        attention_mask = torch.ones_like(input_ids)

        generate_ids = self.text_model.module.generate(
            input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            max_new_tokens=300, do_sample=True, top_k=10, temperature=0.5, suppress_tokens=None
        )
        
        # 去除提示部分
        generate_ids = generate_ids[:, input_ids.shape[1]:-1]  # 移除 EOS 标记

        captions = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        
        import re
        def clean_caption(caption):
            return re.sub(r"[^a-zA-Z0-9\s'\",.]", "", caption)

        captions = [clean_caption(caption).strip() for caption in captions]
        return captions[0] 

    def process(self, item: ImageItem) -> ImageItem:
        caption = self._generate_caption(item.image)
        return ImageItem(item.image, {**item.meta, 'joytag_caption': caption})