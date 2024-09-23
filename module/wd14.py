import os

import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import onnxruntime as ort
import numpy as np
import shutil
from config import WD14MODEL
from PIL import Image


class WD14:
    def __init__(self, wd14model, threshold=0.35, char_threshold=0.85):
        self.wd14model = wd14model
        self.threshold = threshold
        self.char_threshold = char_threshold
        self.download_model_from_huggingface(self.wd14model)
        self.onnx_model = ort.InferenceSession(
            os.path.join(os.path.dirname(os.path.abspath(__file__)).split("module")[0], "model",
                         self.wd14model.split('/')[1], "model.onnx"))

    def download_model_from_huggingface(self, repo_id, model_dir=os.path.join(
        os.path.dirname(os.path.abspath(__file__)).split("module")[0], "model"), model_files=None):
        if model_files is None:
            model_files = ["model.onnx", "selected_tags.csv"]
        model_dir = os.path.join(model_dir, self.wd14model.split('/')[1])
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

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = image.resize((448, 448))  # WD14 model requires 448x448 size
        image_array = np.array(image).astype(np.float32)
        image_array = np.expand_dims(image_array, 0)  # Adds an extra dimension
        return image_array

    def infer_tags(self, image_array):
        if not self.onnx_model:
            raise Exception("ONNX model is not loaded.")

        # Run inference
        outputs = self.onnx_model.run(None, {"input": image_array})
        probabilities = outputs[0][0]
        # Load the tags
        df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)).split("module")[0], "model",
                                      self.wd14model.split('/')[1], "selected_tags.csv"))
        tags = df['name'].tolist()
        tags_category = df['category'].tolist()
        # Filter tags based on the threshold
        result_tags = {"general": [], "character": [], "rating": []}
        for prob, tag, category in zip(probabilities, tags, tags_category):
            if category == 0:
                if prob > self.threshold:
                    result_tags["general"].append((tag, prob))
            elif category == 9:
                if prob > self.threshold:
                    result_tags["rating"].append((tag, prob))
            elif category == 4:
                if prob > self.char_threshold:
                    result_tags["character"].append((tag, prob))
        return result_tags

    def process_image(self, image_path):
        image_array = self.preprocess_image(image_path)
        result_tags = self.infer_tags(image_array)
        return result_tags

    def process_folder(self, folder_path):
        supported_formats = ('.jpg', '.jpeg', '.png', '.webp')
        result = {}
        for root, _, files in os.walk(folder_path):
            processingbar=tqdm(initial=0,unit='it',total=len(files))
            for file in files:
                if file.lower().endswith(supported_formats):
                    image_path = os.path.join(root, file)
                    result[file] = self.process_image(image_path)
                processingbar.update(1)
        return result


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__)).split("module")[0]
    test_image_path = os.path.join(project_root, "test")
    wd14 = WD14(WD14MODEL)
    result = wd14.process_folder(test_image_path)
    print(result)
