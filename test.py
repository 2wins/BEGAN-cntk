
import argparse
import os
import numpy as np

import cntk as C
from cntk import Trainer

from utils import *
from model import *
import imageio


parser = argparse.ArgumentParser()
parser.add_argument('--checkpointPath', default='models/BEGAN_G_100.dnn', help='Checkpoint path. default=models/BEGAN_G_100.dnn')
parser.add_argument('--savePath', default='results', help='Save path. default=results')
parser.add_argument('--num_imgs', type=int, default=100, help='The number of generated images. default=100')

opt = parser.parse_args()


if __name__=='__main__':
    exists_or_mkdir(opt.savePath)

    # load the trained model (generator) and extract parameters
    model = C.load_model(opt.checkpointPath)
    z_dim = model.arguments[0].shape[0]  # noise shape
    img_size = np.sqrt(model[0].shape[0]).astype(np.int)
    X_fake = C.combine([model[0].owner])

    for i in range(opt.num_imgs):
        sample_seed = np.random.uniform(-1, 1, size=(1, z_dim)).astype(np.float32)
        output = np.rint((X_fake.eval({X_fake.arguments[0]: sample_seed}) + 1) * 127.5)
        output = np.clip(output, 0, 255).astype(np.uint8)
        output = np.reshape(output, [img_size, img_size])
        imageio.imwrite(os.path.join(opt.savePath, '{:06d}.png'.format(i)), output)
