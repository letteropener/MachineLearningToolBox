from sklearn import svm

x = [[2,0],[1,1],[2,3]]
y = [0,0,1]

clf = svm.SVC(kernel='linear')
clf.fit(x,y)

print(clf)

# support vectors
print(clf.support_vectors_)

# support vectors indices
print(clf.support_)

# number of support vectors for each class
print(clf.n_support_)

# predict
print(clf.predict(X=[[2,0]]))