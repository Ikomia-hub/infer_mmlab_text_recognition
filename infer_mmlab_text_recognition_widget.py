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

# PyQt GUI framework
from PyQt5.QtWidgets import *
import yaml
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

        # Pretrained or custom training
        self.check_custom_training = pyqtutils.append_check(self.grid_layout, "Custom training",
                                                            self.parameters.model_weight_file != "")
        self.check_custom_training.stateChanged.connect(self.on_check_custom_training_changed)

        # Models
        self.combo_model = pyqtutils.append_combo(self.grid_layout, "Model")
        self.combo_config = pyqtutils.append_combo(self.grid_layout, "Config name")
        self.configs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "textrecog")

        for directory in os.listdir(self.configs_path):
            if os.path.isdir(os.path.join(self.configs_path, directory)) and directory != "_base_":
                self.combo_model.addItem(directory)

        self.combo_model.setCurrentText(self.parameters.model_name)
        self.on_combo_model_changed(self.parameters.model_name)
        self.combo_model.currentTextChanged.connect(self.on_combo_model_changed)

        # Model weights
        self.label_model_path = QLabel("Model path (.pth)")
        self.browse_model = pyqtutils.BrowseFileWidget(path=self.parameters.model_weight_file, tooltip="Select file",
                                                       mode=QFileDialog.ExistingFile)
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(self.label_model_path, row, 0)
        self.grid_layout.addWidget(self.browse_model, row, 1)

        # Model cfg
        self.label_cfg = QLabel("Config file (.py)")
        self.browse_cfg = pyqtutils.BrowseFileWidget(path=self.parameters.config_file, tooltip="Select file",
                                                     mode=QFileDialog.ExistingFile)

        # Hide or show widgets depending on user's choice
        self.combo_model.setEnabled(not self.check_custom_training.isChecked())
        self.label_cfg.setEnabled(self.check_custom_training.isChecked())
        self.label_model_path.setEnabled(self.check_custom_training.isChecked())
        self.browse_cfg.setEnabled(self.check_custom_training.isChecked())
        self.browse_model.setEnabled(self.check_custom_training.isChecked())

        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(self.label_cfg, row, 0)
        self.grid_layout.addWidget(self.browse_cfg, row, 1)

        # Dict file to decode the model output
        self.browse_dict_file = pyqtutils.append_browse_file(self.grid_layout, "Dict file (.txt)",
                                                             self.parameters.dict_file)
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_combo_model_changed(self, model_name):
        if self.combo_model.currentText() != "":
            self.combo_config.clear()
            current_model = self.combo_model.currentText()
            config_names = []
            yaml_file = os.path.join(self.configs_path, current_model, "metafile.yml")
            if os.path.isfile(yaml_file):
                with open(yaml_file, "r") as f:
                    models_list = yaml.load(f, Loader=yaml.FullLoader)['Models']

                available_cfg = [os.path.splitext(os.path.basename(model_dict["Config"]))[0] for model_dict in models_list]
                for experiment_name in available_cfg:
                    self.combo_config.addItem(experiment_name)
                    config_names.append(experiment_name)

                selected_cfg = os.path.splitext(self.parameters.cfg)[0]
                if selected_cfg in config_names:
                    self.combo_config.setCurrentText(selected_cfg)
                else:
                    self.combo_config.setCurrentText(available_cfg[0])

    def on_check_custom_training_changed(self, int):
        self.combo_model.setEnabled(not self.check_custom_training.isChecked())
        self.combo_config.setEnabled(not self.check_custom_training.isChecked())

        self.label_cfg.setEnabled(self.check_custom_training.isChecked())
        self.label_model_path.setEnabled(self.check_custom_training.isChecked())
        self.browse_cfg.setEnabled(self.check_custom_training.isChecked())
        self.browse_model.setEnabled(self.check_custom_training.isChecked())

    def on_apply(self):
        # Apply button clicked slot

        # Get parameters from widget
        # Example : self.parameters.windowSize = self.spinWindowSize.value()
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.model_weight_file = self.browse_model.path
        self.parameters.config_file = self.browse_cfg.path
        self.parameters.cfg = self.combo_config.currentText()+".py"
        self.parameters.dict_file = self.browse_dict_file.path
        # update model
        self.parameters.update = True

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferMmlabTextRecognitionWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_mmlab_text_recognition"

    def create(self, param):
        # Create widget object
        return InferMmlabTextRecognitionWidget(param, None)
