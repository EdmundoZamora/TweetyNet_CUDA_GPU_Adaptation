### Binary classifies bird vocalizations in a wav files

### Stores wav and csv data in data/raw/ it outputs results in data/out/

### Takes in raw wave files, trains and outputs best weights and performance and data evaluation(labeling). 

### Run entire project with:  python run.py data features model evaluate  : deletes data directory and recreates each time the above command is ran.

### If data is already downloaded, spare your self the wait using : python run.py data skip features model evaluate : including skip in the targets skips the data downloading step.