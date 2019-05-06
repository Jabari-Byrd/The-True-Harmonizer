import py_midicsv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# creates a csv string from the midi
csv_string = py_midicsv.midi_to_csv(
    "Music\Rap\Four_five_seconds_-_Rihanna_Paul_McCartney_Kanye_West.mid")

# writes the csv string into a file so that its easier for panda to turn it into a data fram
file = open("csv_string.csv", "w")
for data in csv_string:
    file.write("%s" % data)
file.close()

# creates a data frame from the csv file
midifile = pd.read_csv('csv_string.csv', header=None, names=[
                       'Track', 'Time', 'Note_on_c', 'Channel', 'Note', 'Velocity'], usecols=[0, 1, 2, 3, 4, 5])

# only takes in the information with the " Note_on_c" in it
midifile = midifile[midifile.Note_on_c == ' Note_on_c']

# print(midifile.head())

# split up the data between the treble and bass
X = midifile[midifile.Track != 2]
# print(X.head)
Y = midifile[midifile.Track != 1]


i = 0  # counter used to split up the notes

totalstarttime = 0  # used to find the complete start time of the note group

starttime = 0  # used to find the start time for a specific note
endtime = 0

ThreeNotes = []  # the array of the 3 notes that will be added to the main array later

# This is the array of ThreeNotes arrays.  This is the one that the K-Nearest-Neighbor will be looking at
BigMombaNoteArray = []

# a bunch of code to pick out groups of notes and put it into the threenotes array.
# The features are the note, note length, and note start time
for index, note in X.iterrows():
    if (note['Velocity'] > 0):  # only use notes that have been played, not ones that have been stopped

        # if its the first note of the group, you need to use the notes start time as the total start time too
        if i == 0:

            totalstarttime = note['Time']

            # find when the note has stopped and make the the end time
            for index2, end in X.iterrows():
                if end['Time'] > note['Time']:
                    if end['Note'] == note['Note']:
                        if end['Velocity'] == 0:
                            endtime = end['Time']

            # the start of the array will be the total start time so we can know what range the
            # notes lie and and make it easier to find the bass notes
            ThreeNotes.append(totalstarttime)
            ThreeNotes.append(endtime)

        # if its the final note of the group, you need to use the notes end time as the total end time too.
        elif i == 2:
            starttime = note['Time']  # used to know the length of the note

            # find when the note has stopped and make the the end time and total end time
            for index2, end in X.iterrows():
                if end['Time'] > note['Time']:
                    if end['Note'] == note['Note']:
                        if end['Velocity'] == 0:
                            totalendtime = end['Time']
            ThreeNotes.append(starttime)
            ThreeNotes.append(totalendtime)

        else:
            starttime = note['Time']  # used to know the length of the note

            # find when the note has stopped and make the the end time
            for index2, end in X.iterrows():
                if end['Time'] > note['Time']:
                    if end['Note'] == note['Note']:
                        if end['Velocity'] == 0:
                            endtime = end['Time']
            ThreeNotes.append(starttime)
            ThreeNotes.append(endtime)

        ThreeNotes.append(note['Note'])  # adds note to the ThreeNotes array
        notelength = endtime - starttime  # calculates the length of the note

        # adds the note length calculation to the ThreeNotes array
        ThreeNotes.append(notelength)

        # if i==2 then put the number back down to 0 because we have a full group, else count up.
        # It also appends the totalendtime to the ThreeNotesArray and then emptys that array for
        # the next three notes.  Also add the array to the super huge one.
        if (i == 2):
            i = 0

            # the end of the array will be the total end time so we can know what range the
            # notes lie and and make it easier to find the bass notes
            ThreeNotes.append(totalendtime)

            BigMombaNoteArray.append(ThreeNotes)

            ThreeNotes = []

        else:
            i += 1

print(BigMombaNoteArray[0])

# X_train, X_test, y_train, y_test = train_test_split(X, Y)
