import os
import urllib.request
import tarfile
from zipfile import ZipFile

def get_data(Skip,outdir):
    '''
    download and extract wav and csv data from NIPS4B.
    Handle duplicate data
    '''
    #print(outdir)
    #print(type(outdir))
    #print(os.path.join(outdir,"NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV"))
    if Skip:
        return os.path.join(outdir,"NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV")
    else:    
        wav_dl = "http://sabiod.univ-tln.fr/nips4b/media/birds/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV.tar.gz"
        csv_dl = "https://figshare.com/ndownloader/files/16334603"
        print("----------------------------------------------------------------------------------------------")
        print("\n")
        print("Downloading data files")
        print("\n")
        print("----------------------------------------------------------------------------------------------")
        tar_path, responseHeaders = urllib.request.urlretrieve(wav_dl, os.path.join(outdir, 'NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV.tar.gz'))
        csv_path, responseHeaders = urllib.request.urlretrieve(csv_dl, os.path.join(outdir, 'temporal_annotations_Nips4b.zip'))
        print("----------------------------------------------------------------------------------------------")
        print("\n")
        print("Finished downloading data files")
        print("\n")
        print("----------------------------------------------------------------------------------------------")
        print(tar_path, responseHeaders)
        print(csv_path, responseHeaders)
        
        # extract tar 
        tp = tar_path
        print("\n")
        print("----------------------------------------------------------------------------------------------")
        print("\n")
        print("Extracting wavs from tar")
        print("\n")
        print("----------------------------------------------------------------------------------------------")
        with tarfile.open(tp,'r:gz') as tf: # this part takes a while
            tf.extractall(outdir) # what is the path to this
        print("\n")
        print("----------------------------------------------------------------------------------------------")
        print("\n")
        print("wavs from tar extracted")
        print("\n")
        print("----------------------------------------------------------------------------------------------")

        # extract zip
        cp = csv_path
        print("\n")
        print("----------------------------------------------------------------------------------------------")
        print("\n")
        print("Extracting csv's from zip")
        print("\n")
        print("----------------------------------------------------------------------------------------------")
        with ZipFile(cp) as zf: # this needs to get moved into the NIPS folder, or take out train and test folders
            zf.extractall(os.path.join(outdir,"NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV"))
        print("\n")
        print("----------------------------------------------------------------------------------------------")
        print("\n")
        print("csv's from zip extracted")
        print("\n")
        print("----------------------------------------------------------------------------------------------")
        # remove compressed files
        print("\n")
        print("----------------------------------------------------------------------------------------------")
        print("\n")
        print("removing zip")
        os.remove(cp)
        print("removing tar")
        os.remove(tp)
        print("\n")
        print("----------------------------------------------------------------------------------------------")
        print("\n")
        return os.path.join(outdir,"NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV")

#get_data("C:\\Users\lianl\Repositories\Methodology5\data\\raw")