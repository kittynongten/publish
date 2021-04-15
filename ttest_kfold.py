from sklearn.model_selection import KFold
import numpy as np
X = np.array([[1,2 ], [3, 4], [5, 6], [7, 8]])


kf=KFold(n_splits=2, random_state=None, shuffle=False)
for train_index, test_index in kf.split(X):

    print("TRAIN: ", train_index, "TEST: ", test_index)
    X_train = X[train_index]
    X_test  = X[test_index]
    print("X_train: ",X_train)
    print("X_test : ",X_test)
    print("\n")