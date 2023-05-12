# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import utils, core, dataprocess
import copy
from mmocr.utils import register_all_modules
from mmocr.apis.inferencers import TextRecInferencer
import torch
from infer_mmlab_text_recognition.utils import polygon2bbox, bbox2polygon
from tempfile import NamedTemporaryFile
import os
import cv2
import numpy as np
from mmengine import Config


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferMmlabTextRecognitionParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        # Example : self.windowSize = 25
        self.update = False
        self.model_name = "satrn"
        self.cfg = "satrn_shallow-small_5e_st_mj.py"
        self.weights = "https://download.openmmlab.com/mmocr/textrecog/satrn/satrn_shallow-small_5e_st_mj/satrn_shallow-small_5e_st_mj_20220915_152442-5591bf27.pth"
        self.config_file = ""
        self.model_weight_file = ""
        self.custom_training = False
        self.batch_size = 64
        self.dict_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dicts", "english_digits_symbols.txt")

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(param_map["windowSize"])
        self.update = utils.strtobool(param_map["update"])
        self.model_name = param_map["model_name"]
        self.cfg = param_map["cfg"]
        self.config_file = param_map["config_file"]
        self.weights = param_map["weights"]
        self.model_weight_file = param_map["model_weight_file"]
        self.custom_training = utils.strtobool(param_map["custom_training"])
        self.batch_size = int(param_map["batch_size"])
        self.dict_file = param_map["dict_file"]

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        # Example : paramMap["windowSize"] = str(self.windowSize)
        param_map["update"] = str(self.update)
        param_map["model_name"] = self.model_name
        param_map["cfg"] = self.cfg
        param_map["config_file"] = self.config_file
        param_map["weights"] = self.weights
        param_map["model_weight_file"] = self.model_weight_file
        param_map["custom_training"] = str(self.custom_training)
        param_map["batch_size"] = str(self.batch_size)
        param_map["dict_file"] = self.dict_file
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferMmlabTextRecognition(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here
        # Example :  self.addInput(dataprocess.CImageIO())
        #           self.add_output(dataprocess.CImageIO())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # number of words to recognize per model run

        self.add_output(dataprocess.CTextIO())

        self.model = None
        # Create parameters class
        if param is None:
            self.set_param_object(InferMmlabTextRecognitionParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        param = self.get_param_object()
        batch_size = param.batch_size
        # Get input :
        input = self.get_input(0)
        graphics_input = self.get_input(1)

        img = input.get_image()

        # Get output :
        text_output = self.get_output(1)

        self.forward_input_image(0, 0)

        # Load models into memory
        if self.model is None or param.update:
            print("Loading text recognition model...")
            if self.model is None or param.update:
                if param.model_weight_file != "":
                    param.use_custom_model = True
            if not param.custom_training:
                config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "textrecog",
                                      param.model_name, param.cfg)
                cfg = config
                ckpt = param.weights
            else:
                tmp_cfg = NamedTemporaryFile(suffix='.py')
                cfg = param.config_file
                cfg = Config.fromfile(cfg)
                cfg.model.decoder.dictionary.dict_file = param.dict_file
                cfg.dump(tmp_cfg.name)
                cfg = tmp_cfg.name
                ckpt = param.model_weight_file
            register_all_modules()
            self.model = TextRecInferencer(cfg, ckpt, device=self.device)
            param.update = False
            print("Model loaded!")

        if self.model is not None:
            if img is not None:
                color = [255, 0, 0]

                # Shape of output image
                h_original, w_original, _ = np.shape(img)

                nb_obj = 0
                # Check if there are boxes as input
                if graphics_input.is_data_available():
                    nb_obj = len(graphics_input.get_items())

                if nb_obj > 0:
                    polygons = graphics_input.get_items()
                    imgs = []
                    boxes = []
                    # create batch of images containing text
                    for polygon in polygons:
                        if polygon.is_text_item():
                            continue
                        bbox = polygon.get_bounding_rect()
                        x, y, w, h = [int(coord) for coord in bbox]

                        crop_img = img[y:y + h, x:x + w]
                        if np.cumprod(np.shape(crop_img)).flatten()[-1] > 0:
                            imgs.append(crop_img)
                            boxes.append(bbox)

                    results = self.batch_infer(imgs, batch_size)
                    for i, (box, prediction) in enumerate(zip(boxes[::-1], results[::-1])):
                        text = prediction['text']
                        conf = prediction['scores']
                        box_x, box_y, box_width, box_height = [float(c) for c in box]
                        text_output.add_text_field(id=i, label="", text=text, confidence=float(conf),box_x=box_x, box_y=box_y, box_width=box_width, box_height=box_height, color=color)

                # If there is no box input, the whole image is passed to the model
                else:
                    h, w, _ = np.shape(img)
                    prediction = self.model([img])['predictions'][0]
                    text = prediction['text']
                    conf = prediction['scores']
                    text_output.add_text_field(id=0, label="", text=text, confidence=float(conf), box_x=0., box_y=0., box_width=float(w), box_height=float(h), color=color)

            else:
                print("No input image")
        else:
            print("No model loaded")

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()

    def batch_infer(self, imgs, batch_size):
        chunks = [self.model(imgs[i:i+batch_size], batch_size=batch_size)['predictions'] for i in range(0, len(imgs), batch_size)]
        return [_ for __ in chunks for _ in __]

    def stop(self):
        super().stop()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferMmlabTextRecognitionFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_mmlab_text_recognition"
        self.info.short_description = "Inference for MMOCR from MMLAB text recognition models"
        self.info.description = "If custom training is disabled, models will come from MMLAB's model zoo." \
                                "Else, you can also choose to load a model you trained yourself with our plugin " \
                                "train_mmlab_text_recognition. In this case make sure you give to the plugin" \
                                "a config file (.py) and a model file (.pth). Both of these files are produced " \
                                "by the train plugin."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Text"
        self.info.version = "1.1.1"
        self.info.icon_path = "icons/mmlab.png"
        # self.info.icon_path = "your path to a specific icon"
        self.info.authors = "Kuang, Zhanghui and Sun, Hongbin and Li, Zhizhong and Yue, Xiaoyu and Lin," \
                            " Tsui Hin and Chen, Jianyong and Wei, Huaqiang and Zhu, Yiqin and Gao, Tong and Zhang," \
                            " Wenwei and Chen, Kai and Zhang, Wayne and Lin, Dahua"
        self.info.article = "MMOCR:  A Comprehensive Toolbox for Text Detection, Recognition and Understanding"
        self.info.journal = "Arxiv"
        self.info.year = 2021
        self.info.license = "Apache-2.0 License"
        # URL of documentation
        self.info.documentation_link = "https://mmocr.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/open-mmlab/mmocr"
        # Keywords used for search
        self.info.keywords = "inference, mmlab, mmocr, ocr, text, recognition, pytorch, satrn, seg"

    def create(self, param=None):
        # Create process object
        return InferMmlabTextRecognition(self.info.name, param)
