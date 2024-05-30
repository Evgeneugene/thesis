#import libraries
import numpy as np
import medmnist.dataset
from tsimcne.imagedistortions import *
from tsimcne.tsimcne import TSimCNE
from tsimcne.evaluation.eval import knn_acc,silhouette_score_
from torch.utils.data import ConcatDataset
from matplotlib import pyplot as plt
#load the data
print("BloodMNIST STARTED:")

root='datasets'
dataset_train = medmnist.dataset.BloodMNIST(root=root, split='train', transform=None,target_transform=None, download=True)
dataset_test = medmnist.dataset.BloodMNIST(root=root, split='test', transform=None, target_transform=None, download=True)
dataset_val = medmnist.dataset.BloodMNIST(root=root, split='val', transform=None, target_transform=None, download=True)
dataset_full = [dataset_train, dataset_test,dataset_val]
 
for dataset in dataset_full:
        dataset.labels = dataset.labels.squeeze()
dataset_full_ = ConcatDataset(dataset_full)

labels = np.array([lbl for img, lbl in dataset_full_])


batch_size=1024
total_epochs=[100,20,50]

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
ax.scatter(*Y.T, c=labels, s=1)
ax.set_title(f"$k$NN acc. = {kNN_score}% sil score = {sil_score}")
fig.savefig("tsimcne_bloodmnist_long_auccl.png")

images = []
for img, lbl in dataset_full_:
    images.append(img)  # Convert PyTorch tensor to numpy array if needed
images = np.array(images)

npz_images = np.savez('numpy_files/tsimcne_bloodmnist_long_auccl_all.npz', embeddings=Y, labels=labels, images=images)