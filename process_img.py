import os
import shutil

from config import WD14MODEL, AESTHETIC
from module.wd14 import WD14
from module.aesthetic_rating import Aesthetic
from waifuc.action import FilterSimilarAction,ModeConvertAction
from waifuc.export import SaveExporter
from waifuc.source import LocalSource
from config import RAW_DATA_DIR,PROCESSED_DATA_DIR
from PIL import Image
if __name__ == '__main__':
    project_root = os.path.dirname(os.path.abspath(__file__)).split("module")[0]
    raw_image_path = os.path.join(project_root, RAW_DATA_DIR)
    processed_image_path = os.path.join(project_root, PROCESSED_DATA_DIR)
    rating = Aesthetic(AESTHETIC)
    rating_result = rating.process_folder(raw_image_path)
    for key in rating_result:
        #print(rating_result[key])
        image = Image.open(os.path.join(raw_image_path, key))
        if rating_result[key][1]<0.23:
            image.close()
            low_quality_folder = os.path.join(project_root, "low_quality")
            if not os.path.exists(low_quality_folder):
                os.makedirs(low_quality_folder)
            source_path = os.path.join(raw_image_path, key)
            destination_path = os.path.join(low_quality_folder, key)
        
            if os.path.exists(source_path):
                shutil.move(source_path, destination_path)
                #print(f"Moved {key} to low_quality folder.")
            else:
                print(f"File {key} not found.")
        elif rating_result[key][1] >= 0.23 and ((image.width / image.height) > 4 or (image.height / image.width) > 4):
            image.close()
            high_aspect_ratio_folder = os.path.join(project_root, "high_aspect_ratio")
            if not os.path.exists(high_aspect_ratio_folder):
                os.makedirs(high_aspect_ratio_folder)
            source_path = os.path.join(raw_image_path, key)
            destination_path = os.path.join(high_aspect_ratio_folder, key)
            
            if os.path.exists(source_path):
                shutil.move(source_path, destination_path)
                print(f"Moved {key} to high_aspect_ratio folder.")
            else:
                print(f"File {key} not found.")
        else:
            image.close()
    source = LocalSource(RAW_DATA_DIR)
    source = source.attach(
        ModeConvertAction(mode='RGB', force_background='white'),
        FilterSimilarAction(),
    )
    source.export(SaveExporter(PROCESSED_DATA_DIR))
    wd14 = WD14(WD14MODEL)
    wd14result = wd14.process_folder(processed_image_path)
    for key in wd14result:
        rating_score = rating_result[key][0]
        character_info = [character for character, prob in wd14result[key]['character']]
        tag_file_path_raw = os.path.join(RAW_DATA_DIR, f"{key}.txt")
        if os.path.exists(tag_file_path_raw):
            with open(tag_file_path_raw, 'r', encoding='utf-8') as file:
                original_tags = file.read().splitlines()
        else:
            original_tags = []
        new_tags = [tag for tag, prob in wd14result[key]['general']]
        combined_tags = set(original_tags + new_tags)
        output_lines = [f"{rating_score}"] + [f"{char}" for char in character_info] + list(combined_tags)
        #print(output_lines)
        tag_file_path_test = os.path.join(processed_image_path, f"{key}.txt")
        with open(tag_file_path_test, 'w', encoding='utf-8') as file:
            for line in output_lines:
                file.write(line + ',')