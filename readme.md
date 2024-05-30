#### Installation
$t$-SimCNE is available as a package. You could install it by running 
```python
pip install tsimcne
``` 
or you can clone this repository.
```
git clone https://github.com/berenslab/medical-t-simcne
cd medical-t-simcne
pip install .
```

#### Training a model on MedMNIST dataset
```python
#import libraries
import numpy as np
import medmnist.dataset
from tsimcne.imagedistortions import *
from tsimcne.tsimcne import TSimCNE
from evaluation.eval import knn_acc,silhouette_score_
from torch.utils.data import ConcatDataset

#load the data
root='datasets'
dataset_train = medmnist.dataset.BloodMNIST(root=root, split='train', transform=None,target_transform=None, download=True)
dataset_test = medmnist.dataset.BloodMNIST(root=root, split='test', transform=None, target_transform=None, download=True)
dataset_val = medmnist.dataset.BloodMNIST(root=root, split='val', transform=None, target_transform=None, download=True)
dataset_full = [dataset_train, dataset_test,dataset_val]

for dataset in dataset_full:
        dataset.labels = dataset.labels.squeeze()
dataset_full_ = ConcatDataset(dataset_full)

labels = np.array([lbl for img, lbl in dataset_full_])


batch_size=512
total_epochs=[100,10,40]

# You can also define your custom augmentations by passing a 'data_transform' parameter.
# For more details check scripts/mnist.py or 
# read the documentation here [https://t-simcne.readthedocs.io/]  
tsimcne = TSimCNE(batch_size=batch_size, total_epochs=total_epochs) 
Y = tsimcne.fit_transform(dataset_full_)

#get the metrics
kNN_score=knn_acc(Y,labels)
sil_score=silhouette_score_(Y,labels)

#visualise the results
fig, ax = plt.subplots()
ax.scatter(*Y.T, c=labels)
ax.set_title(f"$k$NN acc. = {kNN_score}% sil score = {sil_score}")
fig.savefig("tsimcne.png")

```
