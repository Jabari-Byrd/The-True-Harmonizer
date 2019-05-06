import py_midicsv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

csv_string = py_midicsv.midi_to_csv(
    "Music\Rap\Four_five_seconds_-_Rihanna_Paul_McCartney_Kanye_West.mid")

file = open("csv_string.csv", "w")
for data in csv_string:
    file.write("%s" % data)
file.close()

information = []
highnotes = []
bassnotes = []

midifile = pd.read_csv('csv_string.csv', header=None, names=[
                       'Track', 'Time', 'Note_on_c', 'Channel', 'Note', 'Velocity'], usecols=[0, 1, 2, 3, 4, 5])

midifile = midifile[midifile.Note_on_c == ' Note_on_c']

midifile = midifile.drop('Note_on_c', axis=1)
midifile = midifile.drop('Channel', axis=1)
midifile = midifile.drop('Velocity', axis=1)
# print(midifile.head())

X = midifile[midifile.Track != 2]
Y = midifile[midifile.Track != 1]
X = X.drop('Track', axis=1)
Y = Y.drop('Track', axis=1)
X = X.drop('Time', axis=1)
Y = Y.drop('Time', axis=1)

X = X.astype('int')
Y = Y.astype('int')

Y = np.resize(Y, X.shape)
# print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y)

scaler = StandardScaler()

scaler.fit(X_train)

StandardScaler(copy=True, with_mean=True, with_std=True)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=.9, beta_2=.999, early_stopping=True, epsilon=1e-08, hidden_layer_sizes=(1000,1000,1000), learning_rate='adaptive',
                    learning_rate_init=0.001, max_iter=50000, momentum=.9, nesterovs_momentum=True, power_t=.5, random_state=None, shuffle=True, solver='adam', tol=.0001, validation_fraction=.1, verbose=False, warm_start=False)

mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))
