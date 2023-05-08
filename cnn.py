import pandas
import numpy

import keras
from keras.models import Sequential
from keras import layers

from sklearn.model_selection import KFold
from sklearn import metrics

# from sklearn.preprocessing import StandardScaler 

dataset = pandas.read_csv("dataset_extended.csv")
dataset = dataset.sample(frac=1)

target = dataset.iloc[:,-1].values
data = dataset.iloc[:,:-1].values
data = data/255.0
# scaler =StandardScaler()
# data = scaler.fit_transform(numpy.array(data, dtype=float))

data = data.reshape(-1,28,28,1)

print(target)
print(data)

split_number = 4
kfold_object = KFold(n_splits=split_number)
kfold_object.get_n_splits(data)


results_accuracy = []
results_confusion = []

for training_index, test_index in kfold_object.split(data):
  data_training = data[training_index]
  target_training = target[training_index]
  data_test = data[test_index]
  target_test = target[test_index]
  

  machine = Sequential()
  machine.add(layers.Conv2D(32, kernel_size=(3, 3),
     activation='relu',
     input_shape=(28, 28, 1)))
  machine.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
  machine.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
  machine.add(layers.MaxPooling2D(pool_size=(2, 2)))
  machine.add(layers.Dropout(0.25))
  
  machine.add(layers.Flatten())
  machine.add(layers.Dense(128, activation='relu'))
  machine.add(layers.Dense(64, activation='relu'))
  machine.add(layers.Dropout(0.25))
  machine.add(layers.Dense(10, activation='softmax'))
  
  machine.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
  
  machine.fit(data_training, target_training, epochs=40, batch_size=128)
  
  new_target = numpy.argmax(machine.predict(data_test), axis=-1)
  
  results_accuracy.append(metrics.accuracy_score(target_test, new_target))
  results_confusion.append(metrics.confusion_matrix(target_test, new_target))
  print(results_accuracy)
  for i in results_confusion:
    print(i)