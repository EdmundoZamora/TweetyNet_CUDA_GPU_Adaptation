import os
import sys
import json 

sys.path.insert(0, 'src')

#region 
# file that gets the data DONE
# function that applies the features DONE
# function that builds the model DONE
# results go into separate files 
# perform on a docker container <-- current challenge CURR
# endregion get cuda to work. works on tweety, slow on gpu if epochs low?

import env_setup
from etl import get_data
from NIPS_training import apply_features, model_build, evaluate

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'features', 'model', 'evaluate'. 
    if data is already downloaded, spare your self the wait using:
    ~
    python run.py data skip features model evaluate
    ~
    including skip in the targets skips the data downloading step.
    `main` runs the targets in order of data=>features=>model=>classifications.
    '''

    if 'data' in targets:
        if "skip" in targets:
            Skip = True
        else:
            Skip = False
            env_setup.make_datadir() # removes each time, handles manually deleting     

        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
        # make the data target
        data = get_data(Skip, **data_cfg)

    if 'features' in targets: # ad to q1-code
        if 'skip' in targets:
            data = os.path.join('data/raw',"NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV")

            with open('config/features-params.json') as fh:
                feats_cfg = json.load(fh)

            all_tags, n_mels, train_dataset, val_dataset, test_dataset, hop_length, sr = apply_features(data, **feats_cfg)
        else:
            with open('config/features-params.json') as fh:
                feats_cfg = json.load(fh)

            all_tags, n_mels, train_dataset, val_dataset, test_dataset, hop_length, sr = apply_features(data, **feats_cfg)


    if 'model' in targets: # get cuda to work, GPU to work. cuda available in tweety env
        if "skip" in targets:
            Skip = True
        else: 
            Skip = False   

        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)
        # make the data target, set outputs to data/temp though, only pickle works, managed to get most, 1 missing
        model, date_str = model_build(all_tags, n_mels, train_dataset, val_dataset, Skip, **model_cfg)
    
    if 'evaluate' in targets:
        with open('config/evaluate-params.json') as fh:
            eval_cfg = json.load(fh)
        # evaluates and stores csvs to out/
        evaluate(model, test_dataset, date_str, hop_length, sr, **eval_cfg)

    return

if __name__ == '__main__':
    # run via:
    # python run.py data features model evaluate
    # or 
    # python run.py data skip features model evaluate
    targets = sys.argv[1:]
    main(targets)
