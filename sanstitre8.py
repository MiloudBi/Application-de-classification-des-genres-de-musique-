# first neural network with keras tutorial
from keras.models import Sequential
from keras.layers import Dense
from sanstitre6 import *
from keras import regularizers
from sklearn.model_selection import train_test_split
import keras
import sklearn as skl
import numpy as np
from extract import compute_feature

from keras.layers import Dropout
from sklearn import preprocessing


#import du données
p=80
x=feature()
y=labeel()
print(x.shape)
x, y = skl.utils.shuffle(x, y, random_state=1)
acp=skl.decomposition.PCA(80)
x = acp.fit_transform(x)
x = preprocessing.normalize(x)

#diviser les données en ensemble d'apprentissage et test
x, xt, y, yt = train_test_split(x, y, test_size=0.1)


print(x.shape)
print(xt.shape)


model = Sequential()
model.add(Dense(400, input_dim=p,  activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(300 , activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(8, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(x, y, epochs=300, batch_size=128)
_, accuracy = model.evaluate(x, y, verbose=1)
print(accuracy)
_, accuracyT = model.evaluate(xt, yt, verbose=1)
print(accuracyT)
#predictions  = model.predict(x)
#model.save("model.h5")
print("Saved model to disk")





"""


scaler = skl.preprocessing.StandardScaler(copy=False)
scaler.fit_transform(x)
scaler.transform(xt)
clf = skl.svm.SVC(kernel='rbf')
clf.fit(x, y)
score = clf.score(xt, yt)
print('Accuracy: {:.2%}'.format(score))

n=len(y)

#model.save("model.h5")
print("Saved model to disk")
from matplotlib import pyplot as plt
eigval = (n-1)/n*acp.explained_variance_
plt.plot(np.arange(1,519),eigval) 

plt.title("Scree plot") 
plt.ylabel("Eigen values") 
plt.xlabel("Factor number") 
plt.show()

plt.plot(np.arange(1,519),np.cumsum(acp.explained_variance_ratio_)) 
plt.title("Explained variance vs. # of factors") 
plt.ylabel("Cumsum explained variance ratio") 
plt.xlabel("Factor number") 
plt.show()



for layer in model.layers:
    #print(layer.get_weights().shape)
    print(layer.get_weights()[0])
  
 """ 