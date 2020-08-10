import pandas as pd
import os
import pickle as pkl
import click


@click.command()
@click.option('--input-path', default='/home/aa/mengdechao/datasets/veriwild')
@click.option('--output-path', default='../outputs/veriwild.pkl')
def veriwild(input_path, output_path):
    PATH = input_path

    images = {}

    images['train']        = open(PATH + '/train_test_split/train_list.txt').read().strip().split('\n')
    images['query_3000']   = open(PATH + '/train_test_split/test_3000_query.txt').read().strip().split('\n')
    images['gallery_3000'] = open(PATH + '/train_test_split/test_3000.txt').read().strip().split('\n')
    images['query_5000']   = open(PATH + '/train_test_split/test_5000_query.txt').read().strip().split('\n')
    images['gallery_5000'] = open(PATH + '/train_test_split/test_5000.txt').read().strip().split('\n')
    images['query_10000']  = open(PATH + '/train_test_split/test_10000_query.txt').read().strip().split('\n')
    images['gallery_10000']= open(PATH + '/train_test_split/test_10000.txt').read().strip().split('\n')

    wild_df = pd.read_csv(f'{PATH}/train_test_split/vehicle_info.txt', sep=';', index_col='id/image')

    # Pandas indexing is very slow, change it to dict
    wild_dict = wild_df.to_dict()
    camid_dict = wild_dict['Camera ID']

    outputs = {}
    for key, lists in images.items():
        output = []
        for img_name in lists:
            item = {
                "image_path": f"{PATH}/images/{img_name}.jpg",
                "name": img_name,
                "id": img_name.split('/')[0],
    #             "cam": wild_df.loc[img_name]['Camera ID'] 
                "cam": camid_dict[img_name]
            }
            output.append(item)
        outputs[key] = output
        
    base_path = os.path.split(output_path)[0]
    if base_path != '' and not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    with open(output_path, 'wb') as f:
        pkl.dump(outputs, f)


if __name__ == "__main__":
    veriwild()