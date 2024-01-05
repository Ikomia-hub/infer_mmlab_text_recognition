<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_mmlab_text_recognition/main/icons/mmlab.png" alt="Algorithm icon">
  <h1 align="center">infer_mmlab_text_recognition</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_mmlab_text_recognition">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_mmlab_text_recognition">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_mmlab_text_recognition/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_mmlab_text_recognition.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Run text recognition algorithms from MMLAB framework. This algorithm will often be applied after a text detection algorithm. You can use ***infer_mmlab_text_detection*** from Ikomia HUB for this task.

Models will come from MMLAB's model zoo if custom training is disabled. If not, you can choose to load your model trained with algorithm *train_mmlab_detection* from Ikomia HUB. In this case, make sure to set parameters for config file (.py) and model file (.pth). Both of these files are produced by the train algorithm.

![Example image](https://raw.githubusercontent.com/Ikomia-hub/infer_mmlab_text_recognition/main/images/billboard-result.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add text detection algorithm
text_det = wf.add_task(name="infer_mmlab_text_detection", auto_connect=True)

# Add text recognition algorithm
text_rec = wf.add_task(name="infer_mmlab_text_recognition", auto_connect=True)

# Run the workflow on image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-hub/infer_mmlab_text_recognition/main/images/billboard.jpg")

# Display results
img_output = text_rec.get_output(0)
recognition_output = text_rec.get_output(1)
display(img_output.get_image_with_mask_and_graphics(recognition_output), title="MMLAB text recognition")
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add text detection algorithm
text_det = wf.add_task(name="infer_mmlab_text_detection", auto_connect=True)

# Add text recognition algorithm
text_rec = wf.add_task(name="infer_mmlab_text_recognition", auto_connect=True)

text_rec.set_parameters({
    "model_name": "satrn",
    "cfg": "satrn_shallow-small_5e_st_mj.py",
    "config_file": "",
    "model_weight_file": "",
    "batch_size": "64",
    "dict_file": "dicts/english_digits_symbols.txt",
})

# Run the workflow on image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-hub/infer_mmlab_text_recognition/main/images/billboard.jpg")
```
- **model_name** (str, default="satrn"): model name. 
- **cfg** (str, default="satrn_shallow-small_5e_st_mj"): name of the model configuration file.
- **conf_thres** (float, default=0.5): object detection confidence.
- **config_file** (str, default=""): path to model config file (only if *custom_training=True*). The file is generated at the end of a custom training. Use algorithm ***train_mmlab_text_recognition*** from Ikomia HUB to train custom model.
- **model_weight_file** (str, default=""): path to model weights file (.pt) (only if *custom_training=True*). The file is generated at the end of a custom training.
- **batch_size** (int, default=64): batch processing to speed up inference time.
- **dict_file** (str, default="dicts/english_digits_symbols.txt"): characters dictionary.

MMLab framework for text recognition offers a large range of models. To ease the choice of couple (model_name/cfg), you can call the function *get_model_zoo()* to get a list of possible values.

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add text recognition algorithm
text_rec = wf.add_task(name="infer_mmlab_text_recognition", auto_connect=True)

# Get list of possible models (model_name, model_config)
print(text_rec.get_model_zoo())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add text detection algorithm
text_det = wf.add_task(name="infer_mmlab_text_detection", auto_connect=True)

# Add text recognition algorithm
text_rec = wf.add_task(name="infer_mmlab_text_recognition", auto_connect=True)

# Run the workflow on image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-hub/infer_mmlab_text_recognition/main/images/billboard.jpg")

# Iterate over outputs
for output in text_rec.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

MMLab text recognition algorithm generates 2 outputs:

1. Forwarded original image (CImageIO)
2. Text detection output (CTextIO)