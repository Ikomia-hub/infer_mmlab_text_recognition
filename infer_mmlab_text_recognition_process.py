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

import cv2
import torch
from ikomia import utils, core, dataprocess
#from mmocr.apis.inferencers import TextDetInferencer
import numpy as np
import copy
import os
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
from infer_mmlab_text_recognition.utils import polygon2bbox, bbox2polygon

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferMmlabTextRecognitionParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.update = False
        self.weights = ""
        self.model_cfg = "satrn/satrn_shallow-small_5e_st_mj.py"
        self.deploy_cfg = "text-recognition/text-recognition_onnxruntime_dynamic.py"
        self.batch_size = 64

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.update = utils.strtobool(param_map["update"])
        self.weights = param_map["weights"]
        self.model_cfg = param_map["model_cfg"]
        self.deploy_cfg = param_map["deploy_cfg"]
        self.batch_size = param_map["batch_size"]

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["update"] = str(self.update)
        param_map["weights"] = str(self.weights)
        param_map["model_cfg"] = self.model_cfg
        param_map["deploy_cfg"] = self.deploy_cfg
        param_map["batch_size"] = self.batch_size
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
        self.device = "cpu"
        # number of words to recognize per model run

        self.addOutput(dataprocess.CGraphicsOutput())
        # Add numeric output
        self.addOutput(dataprocess.CNumericIO())
        self.addOutput(dataprocess.CImageIO())

        self.model_cfg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        "mmocr", "configs","textrecog")
        self.deploy_cfg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        "mmdeploy", "configs", "mmocr")
        self.model = None
        # Create parameters class
        if param is None:
            self.setParam(InferMmlabTextRecognitionParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        param = self.getParam()
        batch_size = param.batch_size
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

        # Load models into memory if needed
        if self.model is None or param.update:
            # Get config files and model path
            model_cfg = os.path.join(self.model_cfg_path, param.model_cfg)
            deploy_cfg = os.path.join(self.deploy_cfg_path, param.deploy_cfg)
            backend_files = [param.weights]

            # read deploy_cfg and model_cfg
            deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
            # build task
            self.task_processor = build_task_processor(model_cfg, deploy_cfg, self.device)
            # process input image and backend model
            self.model = self.task_processor.build_backend_model(backend_files)
            print("Model loaded!")
            # process input image
            self.input_shape = get_input_shape(deploy_cfg)
            param.update = False

        if self.model is not None:
            if img is not None:
                scores = []
                labels_to_display = []
                texts = []
                confidences = []

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

                    # split images into batches
                    chunks = [self.task_processor.create_input(imgs[i:i+batch_size]) for i in range(0, len(imgs), batch_size)]
                    results = []
                    for chunk in chunks:
                    # do model inference
                        with torch.no_grad():
                            out = self.model.test_step(chunk[0])
                            results.append(out)

                    # get text and confidence from results
                    for i in range(len(results)):
                        for l in range(len(results[i])):
                            texts.append(results[i][l].pred_text.item)
                            confidences.append(results[i][l].pred_text.score)

                    for box, score, text in zip(boxes[::-1], confidences, texts):
                        pts = bbox2polygon(box)
                        pts = [core.CPointF(x, y) for x, y in zip(pts[0::2], pts[1::2])]
                        prop_poly = core.GraphicsPolygonProperty()
                        prop_poly.pen_color = [255, 0, 0]
                        graphics_box = graphics_output.addPolygon(pts, prop_poly)
                        graphics_box.setCategory(text)

                        # draw predicted text on an image
                        self.draw_text(to_display, text, box)

                        if isinstance(score, list):
                            # create list of displayed values : confidence for each word and
                            # confidence for eachcharacter
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
                    model_inputs, _ = self.task_processor.create_input(img, self.input_shape)
                    results = self.model.test_step(model_inputs)

                    confidences = results[0].pred_text.score
                    text = results[0].pred_text.item
                    # draw predicted text on an image
                    self.draw_text(to_display, text, [0, 0, w, h])

                    if isinstance(scores, list):
                        # create list of displayed values : confidence for each word and
                        # confidence for each character
                        labels_to_display.append("[" + text + "]")
                        scores.append(np.mean(confidences))
                        for c, s in zip(text, scores):
                            labels_to_display.append(c)
                            scores.append(float(s))
                    else:
                        labels_to_display.append(text)
                        scores.append(confidences)

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
        self.info.shortDescription = "Inference for MMOCR from MMLAB text recognition models in .onnx format"
        self.info.description = "Models should be in .onnx format. Make sure you give to the plugin the" \
                                "corresponding model config file (.py) and deploy config file (.py)." \
                                "If a costum (non-listed) config file is used, it should saved in the" \
                                "appropriate config folder of the plugin."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Text"
        self.info.version = "1.0.1"
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
