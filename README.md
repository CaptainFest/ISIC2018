# ISIC2018
International Skin Imaging Collaboration (ISIC) 2018 Competition

This algorithm compares vanila neural networks in multi-classification task using segmentation masks as additional chanels with using just image information.

## Data preparation
my_preprocess.py module makes from original images and masks 

1. Run shell prompt from root of the project.
2. Execute command to make the script runnable for you
'''
chmod u+x bash_scripts/prepare.sh
'''
2. run prepare sh with following parameters:
   1. */path/to/images/*
   2. */path/to/masks/*
   3. */path/to/save/*
   4. *resize_size*
   5. *number_of_parallel_jobs*
'''
. bash_scripts/prepare.sh /home/irek/My_work/train/data/ /home/irek/My_work/train/binary/ /home/irek/My_work/train/ttt/ 224 12
'''
## Experiments



## Tasks
- [x] Refactor code
- [x] Add bash scripts
- [x] Add readmi with explanations
- [ ] define thesis 
