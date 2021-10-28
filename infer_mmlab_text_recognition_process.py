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

from ikomia import core, dataprocess
import copy
import distutils
from mmcv import Config
from mmocr.apis.inference import disable_text_recog_aug_test
import torch
from mmocr.apis.inference import *
from infer_mmlab_text_recognition.utils import textrecog_models, polygon2bbox, bbox2polygon
from mmcv.runner import load_checkpoint
import os
import cv2
import numpy as np


# Your imports below


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
        self.model_name = "SATRN_sm"
        self.cfg = ""
        self.weights = ""
        self.custom_training = False

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(param_map["windowSize"])
        self.update = distutils.util.strtobool(param_map["update"])
        self.model_name = param_map["model_name"]
        self.cfg = param_map["cfg"]
        self.weights = param_map["weights"]
        self.custom_training = distutils.util.strtobool(param_map["custom_training"])

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        # Example : paramMap["windowSize"] = str(self.windowSize)
        param_map["update"] = str(self.update)
        param_map["model_name"] = self.model_name
        param_map["cfg"] = self.cfg
        param_map["weights"] = self.weights
        param_map["custom_training"] = str(self.custom_training)
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
        #           self.addOutput(dataprocess.CImageIO())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.addOutput(dataprocess.CGraphicsOutput())
        # Add numeric output
        self.addOutput(dataprocess.CNumericIO())
        self.addOutput(dataprocess.CImageIO())

        self.model = None
        # Create parameters class
        if param is None:
            self.setParam(InferMmlabTextRecognitionParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        param = self.getParam()
        # Get input :
        input = self.getInput(0)
        graphics_input = self.getInput(1)

        img = input.getImage()

        # Get output :
        graphics_output = self.getOutput(1)
        drawn_text_output = self.getOutput(3)

        # Init numeric output
        numeric_output = self.getOutput(2)
        graphics_output.setNewLayer("mmlab_text_recognition")
        graphics_output.setImageIndex(0)
        numeric_output.clearData()
        numeric_output.setOutputType(dataprocess.NumericOutputType.TABLE)
        self.forwardInputImage(0, 0)

        config = param.cfg if param.cfg != "" and param.custom_training else None
        ckpt = param.weights if param.weights != "" and param.custom_training else None

        # Load models into memory
        if self.model is None or param.update:
            print("Loading text recognition model...")
            if not (param.custom_training):
                cfg = Config.fromfile(os.path.join(os.path.dirname(__file__), "configs/textrecog",
                                                   textrecog_models[param.model_name]["config"]))
                cfg = disable_text_recog_aug_test(cfg)
                device = torch.device(self.device)
                ckpt = os.path.join('https://download.openmmlab.com/mmocr/textrecog/',
                                    textrecog_models[param.model_name]["ckpt"])
                self.model = init_detector(cfg, ckpt, device=device)
            else:
                cfg = Config.fromfile(config)
                cfg = disable_text_recog_aug_test(cfg)
                device = torch.device(self.device)
                ckpt = ckpt
                self.model = init_detector(cfg, ckpt, device=device)
            param.update = False
            print("Model loaded!")

        if self.model is not None:
            if img is not None:
                # Shape of output image
                h_original, w_original, _ = np.shape(img)
                scores = []
                labels_to_display = []

                # Check if there are boxes as input
                if graphics_input.isDataAvailable():
                    polygons = graphics_input.getItems()
                    to_display = np.zeros_like(img)
                    to_display.fill(255)
                    imgs = []
                    boxes = []

                    # create batch of images containing text
                    for polygon in polygons:
                        pts = polygon.points
                        pts = np.array([[pt.x, pt.y] for pt in pts])
                        x, y, w, h = polygon2bbox(pts)
                        crop_img = img[y:y + h, x:x + w]
                        if np.cumprod(np.shape(crop_img)).flatten()[-1] > 0:
                            imgs.append(crop_img)
                            boxes.append([x, y, w, h])
                    results = self.infere(imgs)

                    for box, prediction in zip(boxes[::-1], results[::-1]):
                        text = prediction['text']
                        score = prediction['score']
                        pts = bbox2polygon(box)
                        pts = [core.CPointF(x, y) for x, y in zip(pts[0::2], pts[1::2])]
                        prop_poly = core.GraphicsPolygonProperty()
                        prop_poly.pen_color = [255, 0, 0]
                        graphics_box = graphics_output.addPolygon(pts, prop_poly)
                        graphics_box.setCategory(text)

                        # draw predicted text on an image
                        self.draw_text(to_display, text, box)

                        if isinstance(score, list):
                            # create list of displayed values : confidence for each word and confidence for each
                            # character
                            labels_to_display.append("[" + text + "]")
                            scores.append(np.mean(score))
                            for c, s in zip(text, score):
                                labels_to_display.append(c)
                                scores.append(float(s))
                        else:
                            labels_to_display.append(text)
                            scores.append(score)

                # If there is no box input, the whole image is passed to the model
                else:
                    to_display = np.zeros_like(img)
                    to_display.fill(255)
                    h, w, _ = np.shape(img)
                    prediction = self.infere([img])[0]

                    # draw predicted text on an image
                    self.draw_text(to_display, prediction['text'], [0, 0, w, h])

                    if isinstance(prediction['score'], list):
                        # create list of displayed values : confidence for each word and confidence for each character
                        labels_to_display.append("[" + prediction['text'] + "]")
                        scores.append(np.mean(prediction['score']))
                        for c, s in zip(prediction['text'], prediction['score']):
                            labels_to_display.append(c)
                            scores.append(float(s))
                    else:
                        labels_to_display.append(prediction['text'])
                        scores.append(prediction['score'])

                # display numeric values
                numeric_output.addValueList(scores, "score", labels_to_display)

                # display drawn image with text prediction
                drawn_text_output.setImage(to_display)

            else:
                print("No input image")
        else:
            print("No model loaded")

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def infere(self, imgs):

        try:
            out = model_inference(self.model,
                                  imgs,
                                  ann=None,
                                  batch_mode=True,
                                  return_data=False)
        except:
            out = []
            for img in imgs:
                out.append(model_inference(self.model,
                                           img,
                                           ann=None,
                                           batch_mode=False,
                                           return_data=False))
        return out

    def draw_text(self, img_display, text, box):
        color = [0, 0, 0]
        x_b, y_b, w_b, h_b = box
        font = cv2.FONT_HERSHEY_SIMPLEX
        (w_t, h_t), _ = cv2.getTextSize(text, fontFace=font, fontScale=1, thickness=1)
        fontscale = w_b / w_t
        org = (x_b, y_b + int((h_b + h_t * fontscale) / 2))
        cv2.putText(img_display, text, org, font, fontScale=fontscale, color=color, thickness=1)

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
        self.info.shortDescription = "Inference for MMOCR from MMLAB text recognition models"
        self.info.description = "If custom training is disabled, models will come from MMLAB's model zoo." \
                                "Else, you can also choose to load a model you trained yourself with our plugin " \
                                "train_mmlab_text_recognition. In this case make sure you give to the plugin" \
                                "a config file (.py) and a model file (.pth). Both of these files are produced " \
                                "by the train plugin."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Text"
        self.info.version = "1.0.0"
        self.info.iconPath = "icons/mmlab.png"
        # self.info.iconPath = "your path to a specific icon"
        self.info.authors = "Kuang, Zhanghui and Sun, Hongbin and Li, Zhizhong and Yue, Xiaoyu and Lin," \
                            " Tsui Hin and Chen, Jianyong and Wei, Huaqiang and Zhu, Yiqin and Gao, Tong and Zhang," \
                            " Wenwei and Chen, Kai and Zhang, Wayne and Lin, Dahua"
        self.info.article = "MMOCR:  A Comprehensive Toolbox for Text Detection, Recognition and Understanding"
        self.info.journal = "Arxiv"
        self.info.year = 2021
        self.info.license = "Apache-2.0 License"
        # URL of documentation
        self.info.documentationLink = "https://mmocr.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/open-mmlab/mmocr"
        # Keywords used for search
        self.info.keywords = "inference, mmlab, mmocr, ocr, text, recognition, pytorch, satrn, seg"

    def create(self, param=None):
        # Create process object
        return InferMmlabTextRecognition(self.info.name, param)
