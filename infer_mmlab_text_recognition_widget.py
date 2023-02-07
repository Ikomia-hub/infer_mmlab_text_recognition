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
from ikomia.utils import pyqtutils, qtconversion
from infer_mmlab_text_recognition.infer_mmlab_text_recognition_process import InferMmlabTextRecognitionParam
from infer_mmlab_text_detection.utils import Autocomplete
# PyQt GUI framework
from PyQt5.QtWidgets import *
from fnmatch import fnmatch
import os


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferMmlabTextRecognitionWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferMmlabTextRecognitionParam()
        else:
            self.parameters = param
         # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Model weights
        self.browse_model = pyqtutils.append_browse_file(self.grid_layout,
                                                        label="Model path (.onnx)",
                                                        path=self.parameters.weights,
                                                        mode=QFileDialog.ExistingFile)

        # Model cfg
        model_cfg_list = self.getFileList("mmocr","textrecog")
        self.combo_model_cfg = Autocomplete(model_cfg_list,
                                            parent=None,
                                            i=True,
                                            allow_duplicates=False)
        self.label_model_cfg = QLabel("Model config (.py)")
        self.grid_layout.addWidget(self.combo_model_cfg, 2, 2)
        self.grid_layout.addWidget(self.label_model_cfg, 2, 0)
        self.combo_model_cfg.setCurrentText(self.parameters.model_cfg)

        # Deploy cfg
        model_cfg_list = self.getFileList("mmdeploy","text-recognition")

        self.combo_deploy_cfg = Autocomplete(model_cfg_list,
                                             parent=None,
                                             i=True,
                                             allow_duplicates=False)
        self.label_deploy_cfg = QLabel("Deploy config (.py)")
        self.grid_layout.addWidget(self.combo_deploy_cfg, 3, 2)
        self.grid_layout.addWidget(self.label_deploy_cfg, 3, 0)
        self.combo_deploy_cfg.setCurrentText(self.parameters.deploy_cfg)
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.setLayout(layout_ptr)

    def getFileList(self, root_folder, task):
        if task=="textrecog":
            cfg_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        root_folder, "configs", task)
        if task=="text-recognition":
            cfg_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        root_folder, "configs", "mmocr", task)
        root = cfg_dir
        pattern = "*.py"
        cfg_list = []
        for path, _, files in os.walk(root):
            for name in files:
                if fnmatch(name, pattern):
                    full_path = os.path.join(path, name)
                    cfg_list.append(os.sep.join(
                                os.path.normpath(full_path).split(os.sep)[-2:]))
        return cfg_list

    def onApply(self):
        # Apply button clicked slot
        # Get parameters from widget
        self.parameters.weights = self.browse_model.path
        self.parameters.model_cfg = self.combo_model_cfg.currentText()
        self.parameters.deploy_cfg = self.combo_deploy_cfg.currentText()
        # update model
        self.parameters.update = True

        # Send signal to launch the process
        self.emitApply(self.parameters)

    def onApply(self):
        # Apply button clicked slot
        # Get parameters from widget
        self.parameters.weights = self.browse_model.path
        self.parameters.model_cfg = self.combo_model_cfg.currentText()
        self.parameters.deploy_cfg = self.combo_deploy_cfg.currentText()
        # update model
        self.parameters.update = True

        # Send signal to launch the process
        self.emitApply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferMmlabTextRecognitionWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the 
        # #one declared in the process factory class
        self.name = "infer_mmlab_text_recognition"

    def create(self, param):
        # Create widget object
        return InferMmlabTextRecognitionWidget(param, None)
