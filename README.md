# TwinGAN -- Unsupervised Image Translation for Human Portraits

![identity_preservation](docs/images/AX18_TwinGAN-15.png)
![search_engine](docs/images/AX18_TwinGAN-18.png)

## Use Pretrained Model.

For a pretrained model (Human to anime), you can download it [here](https://drive.google.com/open?id=1dXfqAODQxB2uNhyQANtZICAjwhNMWnbl):
For a pretrained model (Human to cat), you can download it [here](https://drive.google.com/open?id=1UJEqlH_1sfdmWs6MXKV4H69NGad0rdUB):

## installation.

For this project you will need a python3 version.

You will need the following package:
    
    sudo apt install python3
    sudo apt install virtualenv
    sudo apt install python3-pip
    sudo apt install python3-tk
    sudo apt install cmake
    
Install rabbit mq:

    sudo apt install rabbitmq-server
 
Prepare your virtualenv:

    virtualenv venv
    . venv/bin/activate
    pip install -r requirements.txt   

If you want to exit your virtualenv:

    deactivate
    
Run the following command to translate the demo inputs.

```
python inference/image_translation_infer.py --model_path="twingan_256/256" --image_hw=256 --input_tensor_name="sources_ph" --output_tensor_name="custom_generated_t_style_source:0" --input_image_path="./demo/inference_input/" --output_image_path="./demo/inference_output/"
```

The `input_image_path` can be either one single image or a path containing images.

## Original git repository

You can found the original project [here](https://github.com/jerryli27/TwinGAN):

## Troubleshooting

If you have any issue regarding the install/deployment of the engine,
first check the version of pip

    pip -V

If it's lower than 8.0 then you must upgrade it

    pip install --upgrade pip

If you have a low version of pip it's probably because you have a low version of virtualenv as well.

Upgrade your virtualenv to at least be at the 15.0 version

    sudo pip install virtualenv --upgrade

If you get an error related to subprocess32, you will need to install python-dev

    sudo apt install python-dev
    
## Disclaimer

This personal project is developed and open sourced when I am working for Google, therefore you see Copyright 2018 Google LLC in each file. This is not an officially supported Google product. See [License](LICENSE) and [Contributing](CONTRIBUTING.md) for more details.
