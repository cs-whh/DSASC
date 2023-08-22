import argparse
from myutil import *
import copy
from data import *
import types
import os



parser = argparse.ArgumentParser(description="use pretraining net work for feature extract")
parser.add_argument("--dataset",
                        # required=True,
                        dest='dataset',
                        choices=('fashion_mnist',
                                 'cifar10',
                                 'stl10',
                                 'cifar100'),
                        help="Dataset to train")
parser.add_argument("--model",
                       choices=('dino_vits8',
                                 'dino_vits16',
                                 'dino_vitb8'),
                       default='dino_vits8')
args = parser.parse_args()

if not os.path.exists(features_save_dir):
    os.mkdir(features_save_dir)

setup_seed(100)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.dataset == 'cifar10':
    all_img = False
    num_clusters = 10
    img_size = 80
elif args.dataset == 'stl10':
    all_img = False
    num_clusters = 10
    img_size = 240
elif args.dataset == 'cifar100':
    all_img = False
    num_clusters = 100
    img_size = 80
elif args.dataset == 'fashion_mnist':
    all_img = False
    num_clusters = 10
    img_size = 80


# model choose
model = load_pretrain_model(args.model, args.dataset)
model.to(device)

# add method get_middle_features to model
def get_middle_feature(self, x):
    x = self.prepare_tokens(x)
    features = []
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        if i in [2, 5, 8]:
            features.append(self.norm(x)[:, 0])

    features.append(self.norm(x)[:, 0])
    return features


model.get_middle_feature = types.MethodType(get_middle_feature, model)

# load image
dataset = load_raw_image(args.dataset, img_size)
dl = DataLoader(dataset, batch_size=100)

# features extract
model.eval()
features = []
y_true = torch.empty((0,))
for i, (X, y) in enumerate(dl):
    if args.dataset!='cifar100' and not all_img and i == 10:
        break
    if args.dataset=='cifar100' and not all_img and i == 30:
        break
    X = X.to(device)
    y_true = torch.cat((y_true, y),dim=0)
    with torch.no_grad():
        ls = model.get_middle_feature(X)
    if features == []:
        features = copy.deepcopy(ls)
    else:
        for j in range(len(features)):
            features[j] = torch.cat((features[j], ls[j]), dim=0)

# 保存特征数据
# torch.save({'data':features, 'label':y_true}, './data/' + args.dataset + '_features_list.pt')
torch.save({'data':features, 'label':y_true}, os.path.join(features_save_dir, args.dataset + features_suffix))

