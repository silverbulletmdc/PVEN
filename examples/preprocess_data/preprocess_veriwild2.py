import os.path as osp
import pickle as pkl
import click


"""
For the veriwild2 test set.
The file tree should be like this

.
├── gallery
│   └── gallery
│       └── gallery_final
├── query
│   └── query
│       └── query_final
└── test_split

"""
@click.command()
@click.option('--input-path', default='/home/aa/mengdechao/datasets/veriwild2')
@click.option('--output-path', default='../outputs/veriwild2.pkl')
def veriwild2(input_path, output_path):
    output = {}

    output["train"] = []
    PATH = input_path
    for name in ['A', 'B', 'All']:
        for phase in ['query', 'gallery']:
            raw_metas = open(PATH + f'/test_split/{name}_{phase}.txt').read().strip().split('\n')
            output_list = [
                {
                'image_path': osp.join(PATH, phase, raw.split(' ')[0]),
                'id': raw.split(' ')[1],
                'cam': 1
                }
                for raw in raw_metas
            ]
            output[f'{phase}_{name}'] = output_list

    with open(output_path, 'wb') as f:
        pkl.dump(output, f)    


if __name__ == "__main__":
    veriwild2()