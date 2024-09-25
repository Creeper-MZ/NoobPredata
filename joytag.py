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
import os

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, TAG_OUT_DIR
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


# Load CLIP
print("Loading CLIP")
clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
clip_model = AutoModel.from_pretrained(CLIP_PATH)
clip_model = clip_model.vision_model
clip_model.eval()
clip_model.requires_grad_(False)
clip_model.to("cuda")
huggingface_hub.login(token="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")
# Tokenizer
print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer,
                                                                PreTrainedTokenizerFast), f"Tokenizer is of type {type(tokenizer)}"

# LLM
print("Loading LLM")
text_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
text_model.eval()
text_model.to("cuda")
# Image Adapter
print("Loading image adapter")
image_adapter = ImageAdapter(clip_model.config.hidden_size, text_model.config.hidden_size)
image_adapter.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH , "image_adapter.pt"), map_location="cuda"))
image_adapter.eval()
image_adapter.to("cuda")

clip_model = DataParallel(clip_model)
image_adapter = DataParallel(image_adapter)
text_model = DataParallel(text_model)

def process_folder(folder_path, batch_size=16):
    supported_formats = ('.jpg', '.jpeg', '.png', '.webp')
    result = {}
    image_batch = []
    file_names = []
    
    for root, _, files in os.walk(folder_path):
        processbar = tqdm(total=len(files), unit='it')
        for file in files:
            if file.lower().endswith(supported_formats):
                image_path = os.path.join(root, file)
                image_batch.append(Image.open(image_path))
                file_names.append(file)
                
                if len(image_batch) == batch_size:
                    captions = stream_chat(image_batch)
                    for fname, caption in zip(file_names, captions):
                        result[fname] = caption
                    image_batch = []
                    file_names = []
                
                processbar.update(1)
        
        # 处理剩余的图片
        if image_batch:
            captions = stream_chat(image_batch)
            for fname, caption in zip(file_names, captions):
                result[fname] = caption
    
    return result

@torch.no_grad()
def stream_chat(input_images,):
    torch.cuda.empty_cache()
    
    # 预处理图片批次
    images = clip_processor(images=input_images, return_tensors='pt', padding=True).pixel_values
    images = images.to('cuda')

    # 标记提示词
    prompt = tokenizer.encode(VLM_PROMPT, return_tensors='pt', padding=False, truncation=False,
                              add_special_tokens=False)
    prompt = prompt.to('cuda')

    # 嵌入图片
    with torch.amp.autocast_mode.autocast('cuda', enabled=True):
        vision_outputs = clip_model(pixel_values=images, output_hidden_states=True)
        image_features = vision_outputs.hidden_states[-2]
        embedded_images = image_adapter(image_features)
        embedded_images = embedded_images.to('cuda')

    # 嵌入提示词
    prompt_embeds = text_model.module.model.embed_tokens(prompt.to('cuda'))
    embedded_bos = text_model.module.model.embed_tokens(
        torch.tensor([[tokenizer.bos_token_id]], device='cuda', dtype=torch.int64))

    # 构建提示
    batch_size = embedded_images.shape[0]
    inputs_embeds = torch.cat([
        embedded_bos.expand(batch_size, -1, -1),
        embedded_images.to(dtype=embedded_bos.dtype),
        prompt_embeds.expand(batch_size, -1, -1),
    ], dim=1)

    input_ids = torch.cat([
        torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).expand(batch_size, -1),
        torch.zeros((batch_size, embedded_images.shape[1]), dtype=torch.long),
        prompt.expand(batch_size, -1),
    ], dim=1).to('cuda')
    attention_mask = torch.ones_like(input_ids)
    #print(text_model.module.generate.__doc__)
    generate_ids = text_model.module.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                       max_new_tokens=300, do_sample=True, top_k=10, temperature=0.5,
                                       suppress_tokens=None)
    
    # 去除提示部分
    generate_ids = generate_ids[:, input_ids.shape[1]:]
    generate_ids = generate_ids[:, :-1]  # 移除 EOS 标记

    captions = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    import re

    def clean_caption(caption):
        # 保留引号，逗号和句号，移除其他符号
        return re.sub(r"[^a-zA-Z0-9\s'\",.]", "", caption)

    captions = [clean_caption(caption) for caption in captions]
    #print([caption.strip() for caption in captions])
    return [caption.strip() for caption in captions]

if __name__ == "__main__":
    print(CHECKPOINT_PATH)
    base_path = os.path.join(project_root)
    data_path = RAW_DATA_DIR
    out_path = TAG_OUT_DIR
    print("开始提示词处理")
    result=process_folder(data_path,batch_size=2)
    print("开始保存提示词")
    for k, v in result.items():
        file = k + ".txt"
        tag_file_path_test = os.path.join(out_path, file)
        with open(tag_file_path_test, 'w', encoding='utf-8') as file:
            file.write(v)
    print("提示词保存完成")