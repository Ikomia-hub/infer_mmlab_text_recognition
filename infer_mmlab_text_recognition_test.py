from ikomia.core import task, ParamMap
import ikomia
import os
import yaml
import cv2
from ikomia.utils.tests import run_for_test


def test(t, data_dict):
    plugins_folder = ikomia.ik_registry.getPluginsDirectory()
    plugin_folder = os.path.join(plugins_folder, "Python", t.name)
    img = cv2.imread(data_dict["images"]["detection"]["text"])[::-1]
    input_img = t.get_input(0)
    input_img.set_image(img)
    configs_path = os.path.join(plugin_folder, "configs", "textrecog")
    # loop on every configs available
    for directory in os.listdir(configs_path):
        if os.path.isdir(os.path.join(configs_path, directory)) and directory != "_base_":
            yaml_file = os.path.join(configs_path, directory, "metafile.yml")
            if os.path.isfile(yaml_file):
                with open(yaml_file, "r") as f:
                    models_list = yaml.load(f, Loader=yaml.FullLoader)['Models']
                for model_dict in models_list:
                    cfg = os.path.basename(model_dict["Config"])
                    ckpt = model_dict["Weights"]
                    params = task.get_parameters(t)
                    params["cfg"] = cfg
                    params["model_weight_file"] = ckpt
                    params["model_name"] = directory
                    # without update = 1, model is not updated between 2 test
                    params["update"] = 1
                    task.set_parameters(t, params)
                    yield run_for_test(t)
