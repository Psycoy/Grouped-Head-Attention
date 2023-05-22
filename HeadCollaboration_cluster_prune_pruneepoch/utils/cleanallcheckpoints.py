import argparse
import os
import shutil

# Delete all checkpoints in the Experimental_Results folder.

project_NAME='headcolab'  # TODO check the project name carefully before you do this! other wise you will not delete anything!!!

Scores = []
for root, dirs, files in os.walk("./Experimental_Results/", topdown=True):
    for dirname in dirs:
        dirpath = os.path.join(root, dirname)
        if "checkpoints_"+project_NAME in dirname:
            shutil.rmtree(dirpath)
        

print("\n All checkpoints are deleted.")