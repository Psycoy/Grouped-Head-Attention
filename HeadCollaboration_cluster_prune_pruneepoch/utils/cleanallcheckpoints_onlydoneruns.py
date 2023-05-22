import argparse
import os
import shutil
from os.path import exists



# Delete all checkpoints in the Experimental_Results folder.

project_NAME='headcolab'  # TODO check the project name carefully before you do this! other wise you will not delete anything!!!

Scores = []
SweepRuns_finished = []
for root, dirs, files in os.walk("./Experimental_Results", topdown=True):
    for dirname in dirs:
        dirpath = os.path.join(root, dirname)
        if exists(f'{dirpath}/generate-test.txt'):
            print(f'generate-test.txt exists! Checkpoints of {dirpath} can be deleted!')
            SweepRuns_finished.append(dirpath)
        
        if "checkpoints_"+project_NAME in dirname:
            for _sname in SweepRuns_finished:
                if _sname in dirpath:
                    print(f"Deleting {dirpath}...")
                    try:
                        shutil.rmtree(dirpath)
                    except Exception as e: print("Error message: ", e)
                    break
        

print("\n All checkpoints are deleted.")