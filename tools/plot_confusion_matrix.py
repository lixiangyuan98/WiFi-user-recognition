import os
import h5py
import itertools
import seaborn as sns
import numpy as np
from matplotlib import colors
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from skimage.transform import resize

# 数据集路径
data = "dataset.h5"

height = 256
width = 2048

hf = h5py.File(data, 'r')
X_test = np.expand_dims(hf.get('X_test'), axis=-1)
classes = np.array(hf.get('labels'))
y_test = np.eye(len(classes))[hf.get('y_test')]
hf.close()

X_test = np.array([resize(X_test[i], (height, width), mode='reflect',
                    anti_aliasing=True) for i in range(X_test.shape[0])])

# 模型文件路径
model = load_model('model.h5', compile=False)
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['acc'])

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):

    plt.figure(figsize=(5,5))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm, annot=True, cbar=False, fmt=".2f", cmap=cmap)

    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)

cnf_matrix = confusion_matrix(np.argmax(y_test, 1), np.argmax(model.predict(X_test), 1))
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, title='Normalized confusion matrix')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("confusion_matrix.png", bbox_inches='tight', dpi=150)
plt.show()

print(model.evaluate(X_test, y_test))
