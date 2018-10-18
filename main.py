
import numpy as np
from numpy.random import seed
import pandas as pd

data = pd.read_csv('train.csv')
data1 = pd.read_csv('test.csv')
X_train = data.iloc[:, 1:].values
y_train = data.iloc[:, 0].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

'''from sklearn.preprocessing import OneHotEncoder
ohenc = OneHotEncoder(categorical_features = [0])
y_train = ohenc.fit_transform([y_train])'''

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, 10)

seed(1)
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

def create_model(neurons = 522):
    classifier = Sequential()
    classifier.add(Dense(units = neurons, kernel_initializer = 'uniform', activation = 'relu', input_dim = 784))
    classifier.add(Dense(units = 10, activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier

model = KerasClassifier(build_fn = create_model, epochs = 20, batch_size = 32, verbose = 50)

from sklearn.model_selection import GridSearchCV
neurons = np.arange(255, 556, 100)
param_grid = dict(neurons = neurons)
scoring = {'crit' : 'accuracy'}
grid_search = GridSearchCV(estimator = model, param_grid = param_grid, scoring = scoring,
                           refit = 'crit', verbose = 300, n_jobs = 32)
grid_result = grid_search.fit(X_train, y_train)

X_test = data1.iloc[:].values
y_pred = classifier.predict(X_test)

label = np.argmax(y_pred, axis = 1)

pd.DataFrame(label).to_csv('sample_submission.csv', index = [1], header = ['Label'])
