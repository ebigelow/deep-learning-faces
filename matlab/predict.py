"""

regression with facial expression features 


X : (sparse) array-like, shape = [n_samples, n_features]
    Data.

y : (sparse) array-like, shape = [n_samples] or [n_samples, n_classes].
    Predicted multi-class targets.

"""
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


topk = 0
vec_type = 'multi_hot'
f1 = open('sample_submission.csv')   # TODO: two sample submissions
f2 = open('test.csv')
f2.next()


# Load Data
X = []
Y = []
for line1, line2 in zip(f1.readlines(), f2.readlines()):
    x = np.array([float(i) for i in line1.split(',')])
    X.append(X)

    l2 = line2.split(',')
    y = np.fromstring(l2[0])
    Y.append(y)

    

X = np.vstack(X)
Y = np.vstack(Y)

n, f = X.shape
h = n / 7

X_train = X[h:,:]
X_test  = X[:h,:] 
Y_train = Y[h:,:]
Y_test  = Y[:h,:] 


# Linear regression classifier
classif = OneVsRestClassifier(SVC(kernel='linear'))
classif.fit(X_train, Y_train)

Y_pred = classif.predict(X_test)



total = 0.0
hits  = 0.0
avg_prec = 0.0

for ih in range(h):
    y_truth   = Y_test[ih,:]
    y_predict = Y_pred[ih,:]

    if vec_type in ['one_hot', 'cluster']:
        l_predict = y_predict.argmax()
        l_truth   = y_truth.argmax()

        if topk:
            Yp = [i for i,v in sorted(enumerate(y_predict), key=lambda x: -x[1])][:topk]
            if (l_truth in Yp):
                hits += 1
        elif (l_predict == l_truth):
            hits += 1
        total += 1

    elif vec_type == 'multi_hot':
        Yt = [i for i,v in sorted(enumerate(y_truth), key=lambda x: -x[1])]
        rank = lambda y: Yt.index(y)
        Yp = (y_predict >= (y_predict.mean() + y_predict.std())).nonzero()
        avg_prec += (1 / len(y_truth)) * sum(  len([y_ for y_ in Yp if rank(y_) <= rank(y)]) / rank(y) for y in Yp)
        total    += 1

if hits > 0:
    print '{} Accuracy with {}/{} hits'.format(hits/total, hits, total)
else:
    print '{} MAP with {}/{}'.format(avg_prec/total, avg_prec, total)




