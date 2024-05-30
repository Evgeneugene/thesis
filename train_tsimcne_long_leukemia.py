import os
from PIL import Image
import pandas as pd
from tsimcne.imagedistortions import *
from tsimcne.tsimcne import TSimCNE
from tsimcne.evaluation.eval import knn_acc,silhouette_score_
import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np

print("Leukemia STARTED:")

class Leukemia(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_aml = os.path.join(self.image_folder, self.data.iloc[idx]['img'])
        label = self.data.iloc[idx]['labels']
        
        image = Image.open(image_aml).convert("RGB").resize((28,28))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

csv_file = 'datasets/leukemia_labels.csv'
image_folder = 'datasets/AML-Cytomorphology_LMU/'

batch_size=1024
total_epochs=[100,20,50]
dataset = Leukemia(csv_file, image_folder, transform=None)

labels = []
images = []
for img, lbl in dataset:
    labels.append(lbl)
    images.append(img) 
images = np.array(images)
labels = np.array(labels)

tsimcne = TSimCNE(batch_size=batch_size,
                   total_epochs=total_epochs,
                   )


Y = tsimcne.fit_transform(dataset)

kNN_score=knn_acc(Y,labels)
sil_score=silhouette_score_(Y,labels)
print(f"kNN_score: {kNN_score}")
print(f"Silhouette score: {sil_score}")

fig, ax = plt.subplots()
ax.scatter(*Y.T, c=labels)
ax.set_title(f"$k$NN acc. = {kNN_score}% sil score = {sil_score}", fontsize=7)
fig.savefig("figures/leukemia_auccl.png")



npz_images = np.savez('numpy_files/tsimcne_leukemia_long_auccl_all.npz', embeddings=Y, labels=labels, images=images)