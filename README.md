### Binary classifies bird vocalizations in a wav files

### Stores wav and csv data in data/raw/ it outputs results in data/out/

### Takes in raw wave files, trains and outputs best weights and performance and data evaluation(labeling). 

### Run entire project with: python run.py data features model evaluate nips  : deletes data directory and recreates each time the above command is ran. To supress future dependent library version warnings : python -W ignore run.py data features model evaluate nips :

### If data is already downloaded, spare your self the wait using : python run.py data skip features model evaluate nips: including skip in the targets skips the data downloading step. To supress future dependent library version warnings : python -W ignore run.py skip features model evaluate nips :


website: [DSC180-A09-Eco-Acoustic-Event-Detection](https://edmundozamora.github.io/DSC180-A09-Eco-Acoustic-Detection)

link to CPU Model Repository: [CPU Model Repo](https://github.com/EdmundoZamora/Q1-Project-Code)
