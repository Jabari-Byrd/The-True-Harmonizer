import py_midicsv
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import csv
from sklearn.externals import joblib
import pickle


# # creates a csv string from the midi
# csv_string = py_midicsv.midi_to_csv(
#     "Music\Rap\Four_five_seconds_-_Rihanna_Paul_McCartney_Kanye_West.mid")

csv_input = py_midicsv.midi_to_csv("Music\Input\\userinput.mid")

# # writes the csv string into a file so that its easier for panda to turn it into a data fram
# file = open("csv_string.csv", "w")
# for data in csv_string:
#         file.write("%s" % data)
# file.close()

file = open("csv_input.csv", "w")
for data in csv_input:
    file.write("%s" % data)
file.close()

# creates a data frame from the csv file
midifile = pd.read_csv('midi_note_dataset.csv', header=None, names=[
                       'Track', 'Time', 'Note_on_c', 'Channel', 'Note', 'Velocity'], usecols=[0, 1, 2, 3, 4, 5])

# creates a data frame of the user input
input_midifile = pd.read_csv('csv_input.csv', header=None, names=[
    'Track', 'Time', 'Note_on_c', 'Channel', 'Note', 'Velocity', 'Extra'], usecols=[0, 1, 2, 3, 4, 5, 6])

# seperates the important format stuff from the note stuff in the midi file
input_importantstuff = input_midifile[input_midifile.Note_on_c != ' Note_on_c']

# seperates the important track 1 format stuff from the important track 2 format stuff
input_track1stuff = input_importantstuff[input_importantstuff.Track != 2]
input_track2stuff = input_importantstuff[input_importantstuff.Track != 1]

# takes end of file thing out of track 1 since its in a wierd spot
input_track1stuff = input_track1stuff[input_track1stuff.Note_on_c != ' End_of_file']

# saves the end of track thing for track 1 and 2 since it is in a wierd spot in the midi file
input_track1endtrack = input_track1stuff[input_track1stuff.Note_on_c == ' End_track']
input_track2endtrack = input_track2stuff[input_track2stuff.Note_on_c == ' End_track']
input_end_of_file = input_track2stuff[input_track2stuff.Note_on_c == ' End_of_file']

# takes of end of track thing from track 1 and 2 since its in a wierd spot
input_track1stuff = input_track1stuff[input_track1stuff.Note_on_c != ' End_track']
input_track2stuff = input_track2stuff[input_track2stuff.Note_on_c != ' End_track']
input_track2stuff = input_track2stuff[input_track2stuff.Note_on_c != ' End_of_file']

# takes out header from track 2 since its in a wierd spot
input_track2stuff = input_track2stuff[input_track2stuff.Note_on_c != ' Header']


input_midifile = input_midifile[input_midifile.Note_on_c == ' Note_on_c']

# The order of the data bases will be
# [
#   input_track1stuff,
#   input_midifile (before appending),
#   input_track1endtrack,
#   input_track2stuff
#   input_midifile(after append),
#   input_track2endtrack
#   input_end_of_file
# ]


# only takes in the information with the " Note_on_c" in it
midifile = midifile[midifile.Note_on_c == ' Note_on_c']


# split up the data between the treble and bass
X = midifile[midifile.Track != 2]
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
    numberofnotes = 4

    X = X.values

    index = 0

    # a bunch of code to pick out groups of notes and put it into the SetOfNotes array.
    # The features are the note, note length, and note start time
    for noteset in X:
        # only use notes that have been played, not ones that have been stopped
        if (noteset[5] > 0):

            # if its the first note of the group, you need to use the notes start time as the total start time too
            if i == 0:

                totalstarttime = noteset[1]

                # find when the note has stopped and make the the end time
                for endset in X[index:]:
                    if endset[4] == noteset[4]:
                        if endset[5] == 0:
                            endtime = endset[1]
                            break

                # the start of the array will be the total start time so we can know what range the
                # notes lie and and make it easier to find the bass notes
                SetOfNotes.append(totalstarttime)

            # if its the final note of the group, you need to use the notes end time as the total end time too.
            elif i == numberofnotes:
                starttime = noteset[1]  # used to know the length of the note

                # find when the note has stopped and make the the end time and total end time
                for endset in X[index:]:
                    if endset[4] == noteset[4]:
                        if endset[5] == 0:
                            totalendtime = endset[1]
                            break
                SetOfNotes.append(starttime)

            else:
                starttime = noteset[1]  # used to know the length of the note

                # find when the note has stopped and make that the end time
                for endset in X[index:]:
                    if endset[4] == noteset[4]:
                        if endset[5] == 0:
                            endtime = endset[1]
                            break
                SetOfNotes.append(starttime)

            # adds note to the SetOfNotes array
            SetOfNotes.append(int(noteset[4]))
            notelength = endtime - starttime  # calculates the length of the note

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
        index += 1
    return BigMombaNoteArray


def MatchBasses(Y, BigMombaNoteArray):
    i = 0  # counter used to make sure that you dont use more than 3 basses
    BassMambaDamba = []  # the set of bassnotes that match with the treble
    BassSet = []
    Y = Y.values

    # look through all the sets of notes in the BigMombaNoteArray for matching bass notes
    for notesets in BigMombaNoteArray:
        # the start time for the range of particular notes in the set
        startrange = notesets[0]
        # the end time for the range of particular notes in the set
        endrange = notesets[-1]
        endtime = 0
        BassSet = []
        i = 0

        index = 0
        # look through all the bass notes to find a match with the treble set
        for bassSet in Y[index:]:

            # if the tick count of the bass is in the range of the treble set, match it up
            if (int(bassSet[1]) in range(startrange, endrange)):
                if(bassSet[5] > 0):
                    BassSet.append(bassSet)  # add this bass to the Bass Set
                    for bassend in Y[index:]:
                        if (bassend[4] == bassSet[4]):
                            if (bassend[5] == 0):
                                BassSet.append(bassend)
                                break

                    i += 1  # add one to the counter of basses

                # if the basses are at a higher tick count, you have used all your basses you can within rang so break the loop
            if (bassSet[1] > endrange):
                BassMambaDamba.append(BassSet)
                break

        index += 1
    return BassMambaDamba


# The X (train input) converted into just numbers to make it easier for knn
convertedx = convert_to_numbers(X)
# the Y (expected output) converted into the 3 closest bass for each test input
convertedy = MatchBasses(Y, convertedx)
# the test output converted into just numbers to make it easier for knn
convertedinput = convert_to_numbers(input_midifile)

yarray = []  # y arraay used for the predicted indexs of the convertedy array

# loops through the number of indexs for the array and stores them in the yarray
for i in range(len(convertedy)):
    yarray.append(i)

# does all the k nearest neighbor stuff
neighbor = KNeighborsClassifier(weights='distance', algorithm='auto')
neighbor.fit(convertedx, yarray)

# stores the prediction indexs in an array
predictionsindex = neighbor.predict(convertedinput)

predictions = []  # this is where the real predictions are stored

# loops through the prediction index array and stores the correct predictions in it (by using the indexs inside the convertedy array)
for index in predictionsindex:
    predictions.append(convertedy[index])

# this adds that track1 format stuff to the database that we are outputing
for index, note in input_midifile.iterrows():
    input_track1stuff = input_track1stuff.append(note)
input_midifile = input_track1stuff


# his adds the track1 end track stuff to tthe database that we are outputing
for index, note in input_track1endtrack.iterrows():
    input_midifile = input_midifile.append(note)

# adds the track2 format stuff to the output database
for index, note in input_track2stuff.iterrows():
    input_midifile = input_midifile.append(note)

track2endtime = 0  # used to find what tick track2 ends on

# looks through all the convertedinput elements to calculate the distances between the input notes and the prediction notes
for index in range(len(convertedinput)):

    noteset = convertedinput[index]  # picks a specific set of input notes

    # the first note has the initial time spot for the set
    NotesetStart = int(noteset[0])
    if not predictions[index]:
        continue
    # look for the specific predicted output for the specific input
    predictset = predictions[index]

    # looks at the initial time for the specific predicted set
    PredictionsStart = (predictset[0])[1]

    # does simple math find the distance between the two different initial time spots
    distance = PredictionsStart - NotesetStart

    # fixes the times for the predicted set to shift to the right spot for the input and then adds it to the dataframe with the input
    for note in predictset:
        note[1] = (note[1] - distance)

        # keep look for the largest time point to pick what is the end tick for track 2
        if note[1] > track2endtime:
            track2endtime = note[1]
        # pd.DataFrame([note], columns=['Track', 'Time',
        #                             'Note_on_c', 'Channel', 'Note', 'Velocity'])
        input_midifile = input_midifile.append(pd.DataFrame([note], columns=[
                                               'Track', 'Time', 'Note_on_c', 'Channel', 'Note', 'Velocity']), sort=False)

input_midifile = input_midifile.sort_values(['Track', 'Time'])

# fixes the end track time and adds it to the output database
for index, note in input_track2endtrack.iterrows():
    if (note['Note_on_c'] == ' End_track'):
        note.at['Time'] = track2endtime+1
    input_midifile = input_midifile.append(note)

# adds the end of file thing to the end of the output database
for index, note in input_end_of_file.iterrows():
    input_midifile = input_midifile.append(note)

input_midifile.to_csv('test.csv', index=0, header=0, encoding='utf-8')

harmonized = []
nonfloatharm = []

with open('test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', skipinitialspace=True)
    for row in reader:
        harmonized.append(''.join(row))

for x in harmonized:
    x = x.replace(".0", "")
    for i in range(0, 5):
        x = x.replace(",,", ",")
        x = x.replace('\"\"\"', '\"')
    x = x.strip(",")
    nonfloatharm.append(x)

midi_object = py_midicsv.csv_to_midi(nonfloatharm)

with open("harmonized.mid", "wb") as output_file:
    midi_writer = py_midicsv.FileWriter(output_file)
    midi_writer.write(midi_object)
