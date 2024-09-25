import os
import shutil
from typing import Optional

from imgutils.data import load_image
from waifuc.model import ImageItem

from config import WD14MODEL, AESTHETIC
from module.wd14 import WD14
from module.aesthetic_rating import Aesthetic
from module.joytagAction import JoytagAction
from waifuc.action import FilterSimilarAction, TaggingAction
from waifuc.export import SaveExporter
from waifuc.source import LocalSource
from config import RAW_DATA_DIR,PROCESSED_DATA_DIR
from PIL import Image
from waifuc.action import ProcessAction,FilterAction
from waifuc.source.base import BaseDataSource
original_iter = BaseDataSource._iter
class ResolutionFilter(FilterAction):
    def __init__(self, threshold = 4,save = False):
        self.threshold = threshold
        self.save = save
        self.project_root = os.path.dirname(os.path.abspath(__file__)).split("module")[0]
    def check(self, item: ImageItem) -> bool:
        savepath = os.path.join(self.project_root,"data", "resolution_filtered_data")
        if (item.image.width/item.image.height>=self.threshold) or (item.image.height/item.image.height>=self.threshold):
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            item.image.save(os.path.join(savepath, item.meta["filename"]))
        return (item.image.width/item.image.height<self.threshold) and (item.image.height/item.image.height<self.threshold)
class ScoreLabelAction(ProcessAction):
    def __init__(self, model):
        self.rating = Aesthetic(model)
    def process(self, item: ImageItem) -> ImageItem:
        score = self.rating.process_image(item.meta["path"])
        return ImageItem(item.image, {**item.meta, 'score': score[1],'score_label': score[0]})
class ScoreFilterAction(FilterAction):
    def __init__(self, threshold = 0.23, save = False):
        self.threshold = threshold
        self.save=save
        self.project_root = os.path.dirname(os.path.abspath(__file__)).split("module")[0]
    def check(self, item: ImageItem) -> bool:
        #print("run")
        score = item.meta["score"]
        savepath = os.path.join(self.project_root,"data","score_filtered_data")

        if(self.save and score< self.threshold):
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            item.image.save(os.path.join(savepath,item.meta["filename"]))
        #print(score)
        return score >= self.threshold
class TagProcessAction(ProcessAction):
    def __init__(self, model,threshold=0.35, char_threshold=0.85,save_txt=False):
        self.save_txt=save_txt
        self.model = model
        self.threshold = threshold
        self.char_threshold = char_threshold
        self.tagging = WD14(self.model,self.threshold,self.char_threshold)
        self.project_root = os.path.dirname(os.path.abspath(__file__)).split("module")[0]
        self.processed_image_path = os.path.join(project_root, PROCESSED_DATA_DIR)
    def process(self, item: ImageItem) -> ImageItem:
        wd14result = self.tagging.process_image(item.meta["path"])
        #print(wd14result)
        final = ""
        character_info = [character for character, prob in wd14result['character']]
        file = item.meta["filename"]+".txt"
        tag_file_path_raw = os.path.join(RAW_DATA_DIR, file)
        if os.path.exists(tag_file_path_raw):
            with open(tag_file_path_raw, 'r', encoding='utf-8') as file:
                original_tags = file.read().splitlines()
        else:
            original_tags = []
        new_tags = [tag for tag, prob in wd14result['general']]
        combined_tags = set(original_tags + new_tags)
        output_lines = [f"{char}" for char in character_info] + list(combined_tags)
        # print(output_lines)
        for line in output_lines:
            final += line + ','
        if self.save_txt:
            file = item.meta["filename"] + ".txt"
            tag_file_path_test = os.path.join(self.processed_image_path, file)
            with open(tag_file_path_test, 'w', encoding='utf-8') as file:
                    file.write(item.meta["score_label"]+", "+final)
        return ImageItem(item.image, {**item.meta, 'tags': final})
    
if __name__ == '__main__':
    project_root = os.path.dirname(os.path.abspath(__file__)).split("module")[0]
    raw_image_path = os.path.join(project_root, RAW_DATA_DIR)
    processed_image_path = os.path.join(project_root, PROCESSED_DATA_DIR)
    source = LocalSource(RAW_DATA_DIR)
    source = source.attach(
        ResolutionFilter(save=True),
        ScoreLabelAction(AESTHETIC),
        ScoreFilterAction(save=True),
        FilterSimilarAction(),
        TagProcessAction(WD14MODEL),
        #JoytagAction(device="cuda")
    )
    source.export(SaveExporter(PROCESSED_DATA_DIR))