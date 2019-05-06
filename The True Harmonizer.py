import py_midicsv
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


# creates a csv string from the midi
csv_string = py_midicsv.midi_to_csv(
    "Music\Rap\Four_five_seconds_-_Rihanna_Paul_McCartney_Kanye_West.mid")

csv_input = py_midicsv.midi_to_csv("Music\Input\lose_yourself_topline.mid")

# writes the csv string into a file so that its easier for panda to turn it into a data fram
file = open("csv_string.csv", "w")
for data in csv_string:
    file.write("%s" % data)
file.close()

file = open("csv_input.csv", "w")
for data in csv_input:
    file.write("%s" % data)
file.close()

# creates a data frame from the csv file
midifile = pd.read_csv('csv_string.csv', header=None, names=[
                       'Track', 'Time', 'Note_on_c', 'Channel', 'Note', 'Velocity'], usecols=[0, 1, 2, 3, 4, 5])

input_midifile = pd.read_csv('csv_input.csv', header=None, names=[
    'Track', 'Time', 'Note_on_c', 'Channel', 'Note', 'Velocity'], usecols=[0, 1, 2, 3, 4, 5])


input_midifile = input_midifile[input_midifile.Note_on_c == ' Note_on_c']

# print(input_midifile.head())

# only takes in the information with the " Note_on_c" in it
midifile = midifile[midifile.Note_on_c == ' Note_on_c']

# print(midifile.head())

# split up the data between the treble and bass
X = midifile[midifile.Track != 2]
# print(X.head)
Y = midifile[midifile.Track != 1]


def convert_to_numbers(X):
    i = 0  # counter used to split up the notes

    totalstarttime = 0  # used to find the complete start time of the note group

    totalendtime = 0

    starttime = 0  # used to find the start time for a specific note
    endtime = 0

    SetOfNotes = []  # the array of the 3 notes that will be added to the main array later

    # This is the array of SetOfNotes arrays.  This is the one that the K-Nearest-Neighbor will be looking at
    BigMombaNoteArray = []

    # this is the number of notes that the program is looking at.  You can change this so that the program looks at more or less notes
    numberofnotes = 2

    # a bunch of code to pick out groups of notes and put it into the SetOfNotes array.
    # The features are the note, note length, and note start time
    for index, note in X.iterrows():
        # only use notes that have been played, not ones that have been stopped
        if (note['Velocity'] > 0):

            # if its the first note of the group, you need to use the notes start time as the total start time too
            if i == 0:

                totalstarttime = note['Time']

                # find when the note has stopped and make the the end time
                for index2, end in X.iterrows():
                    if index2 > index:
                        if end['Note'] == note['Note']:
                            if end['Velocity'] == 0:
                                endtime = end['Time']
                                break

                # the start of the array will be the total start time so we can know what range the
                # notes lie and and make it easier to find the bass notes
                SetOfNotes.append(totalstarttime)

            # if its the final note of the group, you need to use the notes end time as the total end time too.
            elif i == numberofnotes:
                starttime = note['Time']  # used to know the length of the note

                # find when the note has stopped and make the the end time and total end time
                for index2, end in X.iterrows():
                    if index2 > index:
                        if end['Note'] == note['Note']:
                            if end['Velocity'] == 0:
                                totalendtime = end['Time']
                                break
                SetOfNotes.append(starttime)

            else:
                starttime = note['Time']  # used to know the length of the note

                # find when the note has stopped and make that the end time
                for index2, end in X.iterrows():
                    if index2 > index:
                        if end['Note'] == note['Note']:
                            if end['Velocity'] == 0:
                                endtime = end['Time']
                                break
                SetOfNotes.append(starttime)

            # adds note to the SetOfNotes array
            SetOfNotes.append(int(note['Note']))
            notelength = endtime - starttime  # calculates the length of the note
            # print(endtime)
            # print("hi")

            # adds the note length calculation to the SetOfNotes array

            # if i==2 then put the number back down to 0 because we have a full group, else count up.
            # It also appends the totalendtime to the SetOfNotesArray and then emptys that array for
            # the next three notes.  Also add the array to the super huge one.
            if (i == numberofnotes):
                i = 0

                # the end of the array will be the total end time so we can know what range the
                # notes lie and and make it easier to find the bass notes
                SetOfNotes.append(totalendtime)

                BigMombaNoteArray.append(SetOfNotes)

                SetOfNotes = []

            else:
                SetOfNotes.append(notelength)
                i += 1

    return BigMombaNoteArray


def MatchBasses(Y, BigMombaNoteArray):
    LastBassUsed = 0  # the last bass note that was matched to a treble
    i = 0  # counter used to make sure that you dont use more than 3 basses
    BassMambaDamba = []  # the set of bassnotes that match with the treble

    BassSet = []

    # print(len(BigMombaNoteArray))

    # look through all the sets of notes in the BigMombaNoteArray for matching bass notes
    for notesets in BigMombaNoteArray:
        # print(notesets)
        # the start time for the range of particular notes in the set
        startrange = notesets[0]
        # the end time for the range of particular notes in the set
        endrange = notesets[-1]
        endtime = 0
        BassSet = []
        i = 0

        # look through all the bass notes to find a match with the treble set
        for index2, basses in Y.iterrows():

            # checks to make sure that the bass you are looking at is not repeated
            if (index2 > LastBassUsed):

                # if the tick count of the bass is in the range of the treble set, match it up
                if (int(basses['Time']) in range(startrange, endrange)):

                    LastBassUsed = index2  # make this the last used bass note
                    # print("hi")
                    BassSet.append(basses)  # add this bass to the Bass Set
                    for index3, bass in Y.iterrows():
                        if (index3 > index2):
                            if (bass['Note'] == basses['Note']):
                                if (bass['Velocity'] == basses['Velocity']):
                                    endtime = bass['Time']
                                    BassSet.append(endtime)
                                    break

                    i += 1  # add one to the counter of basses

                    # if there is 3 basses in a set, break the loop
                    if (i == 3):
                        # print(notesets)
                        # print(len(BassSet))
                        BassMambaDamba.append(BassSet)
                        break

                # if the basses are at a higher tick count, you have used all your basses you can within rang so break the loop
                elif (basses['Time'] > endrange):
                    # print(notesets)
                    BassMambaDamba.append(BassSet)
                    break

                # else you still have more to search through
                else:
                    continue

    return BassMambaDamba


convertedx = convert_to_numbers(X)
convertedy = MatchBasses(Y, convertedx)
convertedinput = convert_to_numbers(input_midifile)

# print(convertedinput)

yarray = []
for i in range(len(convertedy)):
    yarray.append(i)

# print(convertedx)

neighbor = KNeighborsClassifier(weights='distance', algorithm='auto')
neighbor.fit(convertedx, yarray)
predictionsindex = neighbor.predict(convertedinput)
predictions = []

for index in predictionsindex:
    predictions.append(convertedy[index])

# print(predictions)

for index in range(len(convertedinput)):
    noteset = convertedinput[index]
    NotesetStart = int(noteset[0])

    predictset = predictions[index]
    # print((predictset[0])['Time'])
    PredictionsStart = (predictset[0])['Time']

    distance = PredictionsStart - NotesetStart

    i = 0
    temp = 0
    output = []
    for note in predictset:
        if (i == 0):
            note.at['Time'] = (note['Time'] - distance)
            print(note)
            temp = note
            output.append(note)
        else:
            temp['Time'] += note
            temp['Velocity'] = 0
            output.append(temp)

print(output)
