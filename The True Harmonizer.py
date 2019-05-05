import py_midicsv
import numpy as np
import pandas as pd

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

midifile=midifile[midifile.Note_on_c==' Note_on_c']
print(midifile.head())

X = midifile[midifile.Track != 2]
Y = midifile[midifile.Track != 1]
print(Y)

