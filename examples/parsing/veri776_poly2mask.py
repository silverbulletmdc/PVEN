"""
Change polys to masks

author: Dechao Meng
email: mengdechaolive@qq.com
"""
from tqdm import tqdm
import json
import cv2
import numpy as np
import argparse
from vehicle_reid_pytorch.utils import mkdir_p


def poly2mask(polys, classes, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for poly, class_ in zip(polys, classes):
        poly = np.array(poly)
        poly[:, 0] *= mask.shape[1]
        poly[:, 1] *= mask.shape[0]
        poly = poly.astype(np.int)
        cv2.fillPoly(mask, [poly], class_)
    return mask


def get_metas_dirty(item):
    nori_id = item['uris'][0]
    shape = item['resources'][0]['size']
    shape = [shape['height'], shape["width"]]
    polys_list = item['results']['polys']
    polys = [poly['poly'] for poly in polys_list]
    classes = [int(poly['attr']['side']) for poly in polys_list]
    return nori_id, shape, polys, classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", default="poly.json")
    parser.add_argument("--output-path", default="veri776_parsing3165")
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        polygons = json.load(f)
    output_path = args.output_path
    mkdir_p(output_path)

    for i, item in tqdm(enumerate(polygons)):
        image_name = item["image_name"]
        shape = item["shape"]
        polys = item["polys"]
        classes = item["classes"]  
        mask = poly2mask(polys, classes, shape)
        print(image_name)
        image_name = image_name.split('/')[1].split('.')[0]
        cv2.imwrite('{}/{}.png'.format(output_path, image_name), mask)


