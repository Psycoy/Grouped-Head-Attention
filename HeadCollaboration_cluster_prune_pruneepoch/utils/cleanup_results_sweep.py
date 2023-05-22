import argparse
import os
import shutil

# For runs in a sweep, delete all other model savings except the best k models (but keep other records such as logging.)

project_NAME = "headcolab"  # TODO change this accordingly

parser = argparse.ArgumentParser()
parser.add_argument("--SweepFolderName", help="The name of the sweep folder.")
parser.add_argument("--k", help="Keeping top k model parameters to save space.")
args = parser.parse_args()

Scores = []
for root, dirs, files in os.walk("./Experimental_Results/" + args.SweepFolderName, topdown=True):
    for dirname in dirs:
        dirpath = os.path.join(root, dirname)
        evaluation_result_path = os.path.join(dirpath, "generate-test.txt")
        try:
            with open(evaluation_result_path, 'r') as f: 
                try:
                    substring2 = f.readline().split('BLEU4 = ')[1]
                    score=''
                    for Char in substring2:
                        if Char!=',':
                            score+=Char
                        else:
                            break     
                    Scores.append(float(score))
                except IndexError:
                    print("Run {} has no test result file (evaluation started but failed), score set as zero.".format(dirname))
                    Scores.append(0)
        except FileNotFoundError:
            print("Run {} has no test result file  (evaluation not yet started), score set as zero.".format(dirname))
            Scores.append(0)
    scoredir = dict(zip(dirs, Scores))
    break

scoredir = dict(sorted(scoredir.items(), reverse=True, key=lambda item: item[1]))
print("scoredir: ", scoredir)

dirstodel = list(scoredir.keys())[int(args.k):]
for dirname in dirstodel:
    dirpath = os.path.join("./Experimental_Results/" + args.SweepFolderName, dirname, "checkpoints_"+project_NAME)
    try:
        shutil.rmtree(dirpath)
        print("Deleted: {}".format(dirpath))
    except OSError as e:
        print("Error: %s : %s, entry skipped." % (dirpath, e.strerror))

print("\nCheckpoints cleaning done.")