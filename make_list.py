import glob
import json
import os.path

from tqdm import tqdm


def make_dense_list():
    root_dir = '/opt/data/common/SeeingThroughFog/SeeingThroughFogCompressedExtracted'

    json_files = glob.glob(os.path.join(root_dir, 'labeltool_labels', '*.json'))
    samples = []
    for file in tqdm(json_files):
        with open(file, 'r') as f:
            data = json.load(f)

        if data['weather']['dense_fog'] and data['daytime']['day']:
            samples.append(os.path.basename(file)[:-5] + '\n')

    with open('splits/dense_fog/dense_fog.txt', 'w') as f:
        f.writelines(samples)

def make_light_list():
    root_dir = '/opt/data/common/SeeingThroughFog/SeeingThroughFogCompressedExtracted'

    json_files = glob.glob(os.path.join(root_dir, 'labeltool_labels', '*.json'))
    samples = []
    for file in tqdm(json_files):
        with open(file, 'r') as f:
            data = json.load(f)

        if data['weather']['light_fog'] and data['daytime']['day']:
            samples.append(os.path.basename(file)[:-5] + '\n')

    with open('splits/dense_fog/light_fog.txt', 'w') as f:
        f.writelines(samples)


if __name__ == '__main__':
    make_dense_list()
    # make_light_list()
