import py_midicsv
import numpy as np
import tensorflow as tf

csv_string = py_midicsv.midi_to_csv(
    "Music\Rap\Four_five_seconds_-_Rihanna_Paul_McCartney_Kanye_West.mid")

information = []
highnotes = []
bassnotes = []

for note in csv_string:
    temp = note.split(',')
    for attribute in temp:
        if attribute == '1':
            for check in temp:
                if check == ' Note_on_c':
                    highnotes.append(temp)
        elif attribute == '2':
            for check in temp:
                if check == ' Note_on_c':
                    bassnotes.append(temp)
        else:
            temp=temp

highnumpynotes = np.array(highnotes)
bassnumpynotes = np.array(bassnotes)
print(highnumpynotes.shape)
print(bassnumpynotes.shape)
print(bassnumpynotes)

LOGDIR = './graphs'

sess = tf.Session()

LEARNING_RATE = 0.01
BATCH_SIZE = 1000
EPOCHS = 10