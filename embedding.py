import torch
from torchvision import transforms
from modules.unet import UNet, UNetReshade
import PIL
from PIL import Image
import argparse
import os.path
from pathlib import Path
import pandas as pd
import glob
import sys
import numpy as np
import pdb
import pickle
from collections import defaultdict



parser = argparse.ArgumentParser(description='Visualize output for a single Task')

parser.add_argument('--task', dest='task', help="normal, depth or reshading")
parser.set_defaults(task='depth')
args = parser.parse_args()

root_dir = './models/'
image_root_dir = './raw/img_align_celeba/'
identity_path = './raw/identity_CelebA.txt'

trans_totensor = transforms.Compose([transforms.Resize(256, interpolation=PIL.Image.BILINEAR),
                                    transforms.CenterCrop(256),
                                    transforms.ToTensor()])
trans_topil = transforms.ToPILImage()

# os.system(f"mkdir -p {args.output_path}")

# get target task and model
target_tasks = ['normal','depth','reshading']
try:
    task_index = target_tasks.index(args.task)
except:
    print("task should be one of the following: normal, depth, reshading")
    sys.exit()
models = [UNet(), UNet(downsample=6, out_channels=1), UNetReshade(downsample=5)]
model = models[task_index]

map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


df = pd.read_csv(identity_path, sep=' ', header=None, index_col=None)

identity_embeddings = defaultdict(list)
for index, row in df.iterrows():
    id = row[1]
    image_path = os.path.join(image_root_dir, row[0])
    img = Image.open(image_path)
    img_tensor = trans_totensor(img)[:3].unsqueeze(0)
    # compute baseline and consistency output
    for type in ['baseline']: #,'consistency']:
        path = root_dir + 'rgb2'+args.task+'_'+type+'.pth'
        model_state_dict = torch.load(path, map_location=map_location)
        model.load_state_dict(model_state_dict)
        model.mid_conv3.register_forward_hook(get_activation('mid_conv3'))
        baseline_output = model(img_tensor)
        embedding = np.array(activation['mid_conv3']).flatten()
        print (embedding.shape)
        identity_embeddings[id].append(embedding) #TODO: save image it came from?
    print (index)
    if index > 100:
        break

with open('embeddings.pkl', 'wb') as fp:
    pickle.dump(identity_embeddings, fp)