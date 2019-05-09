'''
    CS 484 002, Final Project
    Jabari Byrd, Joseph Parrotta

    * midi_to_csv.py:

        * Reads midi files from MIDI_FOLDER inside source directory
        * Writes "note on" information to CSV file, OUTPUT_FILENAME

'''

MIDI_FOLDER = 'midis/'
OUTPUT_FILENAME = "midi_notes.csv"


import os
import py_midicsv

path = os.path.join(os.getcwd(), MIDI_FOLDER)
print(path)

file = open(OUTPUT_FILENAME, "w")

files_skipped = 0
files_read = 0
notes_written = 0
for filename in os.listdir(path):
    print("READING: ",filename)
    file_path = os.path.join(path, filename)
    try:
        csv_string = py_midicsv.midi_to_csv(file_path)

        #print(type(csv_string[0]))

        for data in csv_string:
            # print(data)
            temp = data.split()
            #print(temp)
            if temp[2] == "Note_on_c," and (temp[0] == "1," or temp[0] == "2,"):
                #print(data)
                file.write("%s" % data)
                notes_written += 1

        files_read += 1

    except:
        print("FILE SKIPPED")
        files_skipped += 1

file.close()

print("\nWRITTEN TO: ", OUTPUT_FILENAME)
print("\tFILES SKIPPED: ", files_skipped)
print("\tFILES READ: ", files_read)
print("\tNOTES WRITTEN: ", notes_written)
print("\n")


