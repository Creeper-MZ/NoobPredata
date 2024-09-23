
import os
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoModel
import shutil
from config import AESTHETIC
import torch
from transformers import pipeline
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Aesthetic:
    def __init__(self, model_name):
        self.model_name = model_name
        self.download_model_from_huggingface(self.model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.pipe = pipeline("image-classification", model=self.model_name, device=0)

    def download_model_from_huggingface(self, repo_id, model_dir=os.path.join(
        os.path.dirname(os.path.abspath(__file__)).split("module")[0], "model"), model_files=None):
        if model_files is None:
            model_files = ["model.safetensors"]
        model_dir = os.path.join(model_dir, self.model_name.split('/')[1])
        # Create the model directory (if it does not exist)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Check and download each model file
        for file_name in model_files:
            file_path = os.path.join(model_dir, file_name)
            if not os.path.exists(file_path):
                print(f"Model file {file_name} does not exist, downloading...")
                temp_path = hf_hub_download(repo_id=repo_id, filename=file_name, cache_dir=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)).split("module")[0], "model", "cache"),
                                            force_download=True)
                shutil.move(temp_path, os.path.join(model_dir, file_name))
            else:
                print(f"Model file {file_name} already exists, skipping download.")

        print(f"Model downloaded finished, saved to {model_dir}")




    def process_image(self, image_path):
        single_image_file = image_path  # @param {type:"string"}

        result = self.pipe(images=[single_image_file])
        #print(result)
        prediction_single = result[0]
        result = round([p for p in prediction_single if p['label'] == 'hq'][0]['score'], 2)
        if result>0.71:
            return "very aesthetic",result
        elif 0.71 >= result > 0.45:
            return "aesthetic", result
        elif 0.45 >= result > 0.27:
            return "displeasing", result
        else:
            return "very displeasing", result
    def process_folder(self, folder_path):
        supported_formats = ('.jpg', '.jpeg', '.png', '.webp')
        result = {}
        for root, _, files in os.walk(folder_path):
            processbar=tqdm(total=len(files),unit='it')
            for file in files:
                if file.lower().endswith(supported_formats):
                    image_path = os.path.join(root, file)
                    result[file] = self.process_image(image_path)
                processbar.update(1)
        return result


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__)).split("module")[0]
    test_image_path = os.path.join(project_root, "test")
    rating = Aesthetic(AESTHETIC)
    result = rating.process_folder(test_image_path)
    print(result)