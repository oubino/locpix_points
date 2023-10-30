"""Test pipeline"""

from locpix_points.scripts.preprocess import main as main_preprocess
#from locpix_points.scripts.featextract import main as main_featextract
#from locpix_points.scripts.process import main as main_process
#from locpix_points.scripts.train import main as main_train
import os

def test_pipeline():

    print(os.listdir('.'))

    # run preprocess on data
    main_preprocess(["-i", "../../../../mnt/c/Users/olive/'OneDrive - University of Leeds'/'Research Project'/code/tma/data/raw/locs", "-c", "tests/output/templates/preprocess.yaml", "-o", "tests/output"])

    # run feat extract 

    # run process

    # run train

    # run evaluate


    #assert func(3) == 5
