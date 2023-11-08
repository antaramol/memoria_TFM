import pickle

# get local dir
import os
local_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(local_dir, 'df.pkl'), 'rb') as f:
    df=pickle.load(f)

y_true = df['y_true'].tolist()
y_pred = df['y_pred'].tolist()

labels = list(set(y_true))


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix


# get acuracy
accuracy = sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i]]) / len(y_true)
print('Accuracy: %.3f' % accuracy)

mcm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
print(mcm)

cm = confusion_matrix(y_true, y_pred, labels=labels)
print(cm)


# get f1 score
f1 = f1_score(y_true, y_pred, average='macro')
print('F1 score: %.3f' % f1)


# get precision score
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
print('Precision: %.3f' % precision)

# get recall score
recall = recall_score(y_true, y_pred, average='macro')
print('Recall: %.3f' % recall)


# plot confusion matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

# save all figures into images folder
import os
if not os.path.exists('images'):
    os.makedirs('images')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Label the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
        plt.text(j, i, format(cm[i, j], fmt), fontsize=15,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)

# Plot the confusion matrix
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm, classes=labels, normalize=True, title='Normalized confusion matrix')
# save the figure
# plt.savefig('images/normalized_confusion_matrix.png')
plt.show()


# Plot the confusion matrix
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm, classes=labels, normalize=False, title='Confusion matrix')
# save the figure
# plt.savefig('images/confusion_matrix.png')
plt.show()


# plot multilabel confusion matrix

from sklearn.metrics import ConfusionMatrixDisplay

# 5 subplots for 5 labels
f, axes = plt.subplots(1, len(labels), figsize=(20, 5))
axes = axes.ravel()

for i in range(len(labels)):
    disp = ConfusionMatrixDisplay(confusion_matrix=mcm[i],
                                  display_labels=[0, i])
    disp.plot(ax=axes[i], cmap=plt.cm.Blues, xticks_rotation=90)
    disp.ax_.set_title(labels[i])
    disp.im_.colorbar.remove()

plt.subplots_adjust(wspace=0.5)
f.colorbar(disp.im_, ax=axes)
# save the figure
# plt.savefig('images/multilabel_confusion_matrix.png')
plt.show()



