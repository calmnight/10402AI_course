import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree


def versiontuple(v):
    return tuple(map(int, (v.split("."))))

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    facecolors=cmap(idx),
                    edgecolors='black',
                    marker=markers[idx], 
                    label=f'class {cl}')

    # highlight test samples
    if test_idx is not None:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    facecolors='none',
                    alpha=1.0,
                    edgecolor='black',
                    linewidths=2.5,
                    marker='o',
                    s=120, label='test set')

def main():

    ### Load Iris dataset from scikit-learn
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    print('Class labels:', np.unique(y))


    ### Splitting data into 70% training and 30% test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


    ### Training a Decision Tree via scikit-learn
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
    tree.fit(X_train, y_train)


    ### Model predict
    y_pred = tree.predict(X_test)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())


    ### Model evaluation
    print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))


    ### Visualization
    X_combined_std = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    plot_decision_regions(X_combined_std, y_combined, classifier=tree, test_idx=range(105, 150))
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


    ### Plot tree
    plt.figure(figsize=(12, 8))
    plot_tree(tree,
            feature_names=['petal length', 'petal width'],
            class_names=iris.target_names,
            filled=True,
            rounded=True)
    plt.show()

if __name__ == '__main__':
    main()