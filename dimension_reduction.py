from sklearn.datasets import load_digits
from sklearn.decomposition import *
import matplotlib.pyplot as plt
from sklearn.manifold import *
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

data = load_digits()
print(data.shape)
plt.imshow(data.images[977])


#d_r

pca = PCA(2)
transported_data = pca.fit_transform(data.data)

plt.scatter(transported_data[:,0],transported_data[:,1], c = data.target, cmap = "tab10", s = 4)

#1

def visualize_dim_reduction(name, algorithm, data):
    alg = algorithm(2)
    transported_data = pca.fit_transform(data.data)

    plt.scatter(transported_data[:,0],transported_data[:,1], c = data.target, cmap = "tab10", s = 4)
    plt.title(name)
    plt.show()

dr_algorithms = {"PCA": PCA, "KernelPCA":KernelPCA, "MiniBatchSparsePCA": MiniBatchSparsePCA, "FastICA": FastICA, "LatentDirichletAllocation": LatentDirichletAllocation, "tSNE": TSNE, "MDS": MDS}

for name, algorithm in dr_algorithms.items():
    visualize_dim_reduction(name, algorithm, data)


X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target) #shuffle
cls_algorithms = {"SVC":SVC, "RandomForestClassifier":RandomForestClassifier, "AdaBoostClassifier": AdaBoostClassifier}

for name, algorithm in cls_algorithms.items():
    alg = algorithm()
    alg.fit(X_train, Y_train) #Train
    #y_predicted = alg.predict(X_test) #predict
    ConfusionMatrixDisplay.from_estimator(alg, X_test, Y_test)

