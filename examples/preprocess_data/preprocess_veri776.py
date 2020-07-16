import re
from pathlib import Path
import argparse
import pickle
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", default="veri776.pkl")
    args = parser.parse_args()

    output_dir = os.path.split(args.output_path)[0]
    if output_dir != '' and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    input_path = Path(args.input_path).absolute()
    output_dict = {}

    pattern = re.compile(r"(\d+)_c(\d+)_.+\.jpg")
    for phase in ["train", "query", "gallery"]:
        output_dict[phase] = []
        sub_path = input_path / f"image_{phase}"
        if phase == "gallery":
            sub_path = input_path / f"image_test"
        for image_path in sub_path.iterdir():
            sample = {}
            image_name = image_path.name
            v_id, camera = pattern.match(image_name).groups()
            sample["filename"] = image_name
            sample["image_path"] = str(image_path)
            sample["id"] = v_id
            sample["cam"] = camera
            output_dict[phase].append(sample)
    with open(args.output_path, "wb") as f:
        pickle.dump(output_dict, f)
