#  Latent Outlier Exposure for Anomaly Detection with Contaminated Data (LOE)

The experiments here are an attempt to reproduce the results from https://arxiv.org/abs/2202.08088. and stresstest the NTL https://github.com/boschresearch/NeuTraL-AD. Model. For the original code go to to: https://github.com/boschresearch/LatentOE-AD. This repo contains altered code from the original implementation, with stresstests on robustness and  lowering the amount of data the model is trained on.

Results are only obtained on the fmnist / cifar10 dataset

This Repository is not maintained!

most of the changes to the original repo are in /Additional_CODE_BULE 

 /loe_utils             # contains the helperfunctions for the saving of pickle files in the original LOE implementation
 
 / plots               # plots of the results
 
 /training_scripts      # contains bash script to run multiple runs of the LOE blind etc of ntl
 
 /utils                 # contains helper functions and model for the Added code from the project work
                     
 evaluating_models.ipynb  # compares different models from project work and BOSCH LOE
 inspecting_dataset.ipynb # inspecting the features with dimensionality reduction method and rescaling of the features



## for LOE  

run :
python Launch_Exps.py --config-file config_fmnist.yml   --contamination 0.0 --assumed-contamination 0.0 --dataset-name fmnist  --trainset_fraction

if you want different loss functions etc. change the  config file

- it saves the contaminated results in a different folder, to concatenate them run the function:

concatenate_allresults(MODEL_RESULT_PATH:str,modelname:str='loe_hard',assumed_contamination:float=0.0,n_runs:int=5)
from
LatentOE-AD/Additional_Code_BULE/loe_utils/helperfunctions.py


for the multi-run bash scripts run:

bash Additional_Code_BULE/training_scripts/run_multiple.sh



# Data

- downsample dataset with: Extract_img_features import downsample_dataset on the extraced features


## for project work autencoder etc. 
run training_scripts/ training_main_*.py

- it automatically saves the pickle files of the results in a folder under RESULTS/

each run gets an allresults_run_{}.pkl file which contains all the results for the different contamination ratios



## inspect results

- evaluating_models.ipynb 
