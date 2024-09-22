# taguchi_face_recognition_resnet_v1_openvino_fp16_optimized
face recognition model from dlib-models converted to OpenVINO with FP16 optimization.

## Model
Original model is the 'taguchi_face_recognition_resnet_model_v1.dat' from [dlib-models](https://github.com/davisking/dlib-models) which is also released under CC0 1.0 Universal.  

### How to prepare
1. Convert original model to Tensorflow's saved model by using [dlib-to-tf-keras-converter](https://github.com/ksachdeva/dlib-to-tf-keras-converter).   
You may need to use Ubuntu(WSL) for converter.

2. Convert Tensorflow saved model to OpenVINO by using OpenVINO.  
```python
import openvino as ov
core = ov.Core()
ov_model = core.read_model("./dlib-to-tf-keras-converter/taguchi_face_recognition_resnet_model_v1.pb")
ov.save_model(ov_model, './taguchi_face_recognition_resnet_v1_openvino_fp16_optimized.xml', compress_to_fp16=True) 
```

### Usage
This model has shape ["?,150,150,3"] as below.  
'?' means batch size is not fixed.

```xml
<?xml version="1.0"?>
<net name="TensorFlow_Frontend_IR" version="11">
    <layers>
        <layer id="0" name="input_image:0" type="Parameter" version="opset1">
            <data shape="?,150,150,3" element_type="f32" />
            <output>
                <port id="0" precision="FP32" names="input_image:0">
                    <dim>-1</dim>
                    <dim>150</dim>
                    <dim>150</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
```

When you use it, you may need to set batch size 1 by ```reshape``` option.
```python
import openvino as ov
core = ov.Core()
model_path = "taguchi_face_recognition_resnet_v1_openvino_fp16_optimized.xml"
model = core.read_model(model_path)
model.reshape([1, 150, 150, 3])
```

## Demo
### Setup
1. Install Python 3.12
2. Install libralies by ```pip install -r poetry-requirements.txt```

### Usage
- ```python demo.py```
- ```python demo.py -faces "asset\2_faces_to_be_compared.jpg" -face "asset\2_face_to_compare.jpg"```

#### Option
- ```-faces```: Image which contain faces to be compared to recognize. Default: ```./asset1_faces_to_be_compared.jpg```
- ```-face```: Image which contain a face to compare. Default: ```./asset/1_face_to_compare.jpg```
- ```--tolerance```: Value to judge if face is same or not. This is used for L2 distances. Smaller is strict. Default: ```0.4```
- ```--device```: device to compilze openVINO model. Default: ```CPU```
## LICENSE
For details, please refer to the LICENSE in each folder.
### OpenVINO model:
The OpenVINO model in this repository is licensed under **CC0 1.0 Universal**

- taguchi_face_recognition_resnet_v1_openvino_fp16_optimized.bin  
- taguchi_face_recognition_resnet_v1_openvino_fp16_optimized.xml  

### Code
The code in this repository is licensed under **MIT**

### Images
Images in the asset folder are licended under: **CC0 1.0 Universal**   
The Original images are from "Free Stock Photo".  
Internet shortcuts for each image are sored in the same folder.

## Acknowledgments
I would like to express my gratitude to the following individuals and projects:

[Davis E. King](https://github.com/davisking) and [TAGUCHI Tokuji](https://github.com/TaguchiModels) for releasing the facial recognition model, which have been invaluable for this project.  
[Kapil Sachdeva](https://github.com/ksachdeva) for making the converter publicly available, greatly facilitating the integration process.