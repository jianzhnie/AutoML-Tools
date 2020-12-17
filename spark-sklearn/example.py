from sklearn.utils import parallel_backend
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm
from joblibspark import register_spark

register_spark() # register spark backend

iris = datasets.load_iris()
clf = svm.SVC(kernel='linear', C=1)
with parallel_backend('spark', n_jobs=3):
    scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    print(scores)
