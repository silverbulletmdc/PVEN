# PVEN
This is the official implementation of article "Parsing-based viewaware embedding network for vehicle ReID"[[arxiv]](https://arxiv.org/abs/2004.05021), which has been accpeted by **CVPR20** as a poster article.

## Requirements
1. python 3.6+
2. torch 1.3.1+

## Install
```
git clone https://github.com/silverbulletmdc/PVEN
cd PVEN
pip install -r requirements.txt
python setup.py install
```

If you want to modify the code of this project, use the following commands instead
```
cd PVEN
pip install -r requirements.txt
python setup.py develop
```

## Preparing dataset
Before the pipeline, you should prepare your vehicle ReID dataset first.
For each dataset, you need to generate a description pickle file for it, which is a pickled dict with following structure:
```json
{
    "train":[
        {
            "filename": "0001_c001_00016450_0.jpg",
            "image_path": "/data/datasets/VeRi/VeRi/image_train/0001_c001_00016450_0.jpg",
            "id": "0001",
            "cam": "001",
        },
        ...

    ],
    "gallery":[
        ...
    ],
    "query":[
        ...
    ]
}
```

For different dataset, we have already provided the generating scripts to help you generate the pickle file.
```shell
cd examples/preprocess_data
# For VeRi776
python generate_pkl.py veri776 --input-path <VeRi_PATH> --output-path ../outputs/veri776.pkl
# For VERIWild
python generate_pkl.py veriwild --input-path <VeRi_PATH> --output-path ../outputs/veriwild.pkl
# For VehicleID 
python generate_pkl.py vehicleid --input-path <VeRi_PATH> --output-path ../outputs/vehicleid.pkl
```

## Training the parsing model
<!-- We provide the pre-trained segmentation model on `examples/parsing/best_model_trainval.pth` which you can use to generate parsing masks for different datasets. 
If you want to use the model directly, just skip this section.
At the same time, you can also train your own parsing models follow the following instructions. -->

### Convert polygons to parsing masks
As is described in the article, we annotated the parsing information of 3165 images from VeRi776. 
We just annotate the vertexs of the polygons as the vehicles are composed by several polygons.
The details of polygons are in `examples/parsing/poly.json`.
Run following command to convert the polygons to parsing masks
```
cd examples/parsing
python veri776_poly2mask.py --json-path poly.json --output-path ../outputs/veri776_parsing3165
```
The parsing masks will be generated in `../outputs/veri776_parsing3165` folder.

### Train parsing model

Run following command to train the parsing model
```
cd examples/parsing
python train_parsing.py --train-set trainval --masks-path ../outputs/veri776_parsing3165 --image-path <VeRi_PATH>/image_train
```
where the `<VeRi_PATH>` is the path of your VeRi776 dataset.

## Generate parsing masks for ReID dataset
Running the following command to generate masks for the whole ReID dataset and write the `mask_path` to the dataset pickle file. 
```
cd examples/parsing
python generate_masks.py --model-path best_model_trainval.pth --reid-pkl-path ../outputs/veri776.pkl --output-path ../outputs/veri776_masks
```
where the `<PKL_PATH>` is the generated pickle file above. 

## Train PVEN
Run the following model to train PVEN.
```shell
cd examples/parsing_reid
# For VeRi776
CUDA_VISIBLE_DEVICES=0 python main.py train -c configs/veri776_b64_parsing.yml 
# For vehicleid, use 8 GPUs to train
python main.py train -c configs/vehicleid_b256_pven.yml 
# For VERIWild, use 8 GPUs to train
python main.py train -c configs/veriwild_b256_224_pven.yml 
```

## Pretrained Models
We provide the pretrained parsing model, VeRi776 ReID model and VERIWild ReID model ( the classification layer has been removed ) for your convinient.
You can download it from the following link:
Link: https://pan.baidu.com/s/1Q2NMVfGZPCskh-E6vmy9Cw  password: iiw1

## Evaluate PVEN
```shell
cd examples/parsing_reid
# For VeRi776
python main.py eval -c configs/veri776_b64_parsing.yml

# For VERIWild
## small
python main.py eval -c configs/veriwild_b256_224_pven.yml
## medium
python main.py eval -c configs/veriwild_b256_224_pven.yml test.ext _5000
## Large
python main.py eval -c configs/veriwild_b256_224_pven.yml test.ext _10000
```

## Citation
If you found our method helpful in your research, please cite our work in your publication. 
```bibtex
@inproceedings{meng2020parsing,
  title={Parsing-based View-aware Embedding Network for Vehicle Re-Identification},
  author={Meng, Dechao and Li, Liang and Liu, Xuejing and Li, Yadong and Yang, Shijie and Zha, Zheng-Jun and Gao, Xingyu and Wang, Shuhui and Huang, Qingming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7103--7112},
  year={2020}
}
```
