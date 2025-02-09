from sklearn.datasets import load_digits
from sklearn.decomposition import *
from matplotlib.pyplot as plt

data = load_digits()
print(data.shape)
plt.imshow(data.images[977])


#d_r

pca = PCA(2)
transported_data = pca.fit_transform(data.data)

plt.scatter(transported_data[:,0],transported_data[:,1], c = data.target, cmap = "tab10", s = 4)

