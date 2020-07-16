# PVEN
This is the official code of article "Parsing-based viewaware embedding network for vehicle ReID"[[arxiv]](https://arxiv.org/abs/2004.05021), which has been accpeted by CVPR20 as a poster article.

## Requirements
1. python 3.6+
2. torch 1.3.1+

## Install
Clone this repository. Then excute the following commands
```
cd vehcile_reid.pytorch
pip install -r requirements.txt
python setup.py install
```

If you want to develop this project, use the following commands instead
```
cd vehicle_reid.pytorch
pip install -r requirements.txt
python setup.py develop
```

## Preparing dataset
Before the pipeline, you should preparing your vehicle ReID dataset first.
For each dataset, we need to generate a description pickle file for it, which is a pickled dict with following structure:
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
For different dataset, we provide the generate scripts to help you generate the pickle file.
```shell
cd examples/preprocess_data
# For VeRi776
python preprocess_veri776.py --input-path <VeRi_PATH> --output-path veri776.pkl
# For VehicleID and VeRiWild
# Will be published soon
```


## Training the parsing model
We provide the pre-trained segmentation model on `examples/parsing/best_model_trainval.pth` which you can use to generate parsing masks for different datasets. 
If you want to use the model directly, just skip this section.
At the same time, you can also train your own parsing models follow the following instructions.

### Convert polygons to parsing masks
As is described in the article, we annotate the parsing information of 3165 images from VeRi776. 
We just annotate the vertexs of the polygons as the vehicles are composed by several polygons.
The vertexs information is in `examples/parsing/poly.json`.
Run following command to convert the polygons to parsing masks
```
cd examples/parsing
python veri776_poly2mask.py --input-path poly.json --output-path veri776_parsing3165
```
The parsing masks will be generated in `veri776_parsing3165` folder.

### Train parsing model

Run following command to train the parsing model
```
cd examples/parsing
python train_parsing.py --trainset trainval --masks-path veri776_parsing3165 --image-path <VeRi_PATH>/image_train
```
where the `<VeRi_PATH>` is the path of your VeRi776 dataset.

## Generate parsing masks for ReID dataset
Running the following command to generate masks for the whole ReID dataset and write the `mask_path` to the dataset pickle file. 
```
cd examples/parsing
python generate_masks.py --model-path best_model_trainval.pth --reid-pkl-path <PKL_PATH> --output-path masks
```
where the `<PKL_PATH>` is the generated pickle file above. 

## Train PVEN ReID model
Run the following model to train the PVEN ReID model.
```shell
cd examples/parsing_reid
# For VeRi776
python main.py train -c configs/veri776_b256_parsing.yml data.pkl_path <PKL_PATH>
```
**Note**: To achieve the performance mentioned in paper, you should run the algorithm in the machine equipped with 8 GPUs.

## Evaluating the ReID model
```shell
cd examples/parsing_reid
# For VeRi776
python main.py eval -c configs/veri776_b256_parsing.yml
```

## Citation
If you found our method helpful in your research, please cite our work in your publication. 
```shell
@inproceedings{meng2020parsing,
  title={Parsing-based View-aware Embedding Network for Vehicle Re-Identification},
  author={Meng, Dechao and Li, Liang and Liu, Xuejing and Li, Yadong and Yang, Shijie and Zha, Zheng-Jun and Gao, Xingyu and Wang, Shuhui and Huang, Qingming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7103--7112},
  year={2020}
}
```