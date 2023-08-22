## Deep Structure and Attention aware Subspace Clustering (DSASC)

The DSASC is a subspace clustering method that takes into account the content features and image relationships.
The DSASC uses ViT and GCN to extract image features and perform subspace clustering. Further details can be found in our paper.


### Usage

step 1. Prepare code and environment.
```
git clone git@github.com:cs-whh/DSASC.git
cd DSASC 
pip install -r requirements.txt
```

step 2. Feature extraction using ViT, datasets can be cifar10, stl10, fashion_mnist and cifar100.
```
python feature_extract.py --dataset=cifar10
```

step 3. Train the network and perform clustering.
```
python main.py --dataset=cifar10
```

