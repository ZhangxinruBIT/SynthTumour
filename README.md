# PL-BTS

Our work drew inspiration from the amazing fepegar's [resector](https://github.com/fepegar/resector) tool, which served as the main basis for generating the 3D tumor-like shape. We made certain revisions to the tool to suit our specific needs.
# Installation
```
conda create --name PLBTS python=3.8
conda activate PLBTS
pip install light-the-torch
ltt install torch
pip install git+https://github.com/fepegar/resector
git clone https://github.com/ZhangxinruBIT/PL-BTS.git
pip install -r requirements.txt
```

# Usage for Generation
**Run demo**
```
python resect-multi-tumor.py  DemoData/OAS30003_MR_d1631/flair.nii.gz test_img.nii.gz test_lab.nii.gz

```

To utilize multiprocessing for generating images and their corresponding labels, we need to organize the data in the following structure: 


**Data Structure**


     DemoData
      ├── OAS31157_MR_d4924
      │   ├── flair_brainmask.nii.gz
      │   └── flair.nii.gz
      ├── OAS31158_MR_d2481
      │   ├── flair_brainmask.nii.gz
      │   └── flair.nii.gz
      ├── ...
      │    
      ├── ...
      │   
      ├── OAS31167_MR_d2053
      │   ├── flair_brainmask.nii.gz
      │   └── flair.nii.gz
      ├── ...
    
**Run with multiprocessing**
```
python Prompt-multi-tumor.py

```

After that we could get the generated folds as below:

     Task100_OASIS-PLBTS
      ├── imagesTr
      │   ├── OAS30135_MR_d2931_0000.nii.gz
      │   ├── OAS30137_MR_d3165_0000.nii.gz
      │   ├── OAS30003_MR_d1631task_0000.nii.gz
      │   ├── ...
      ├── mask
      └── labelsTr
      │   ├── OAS30135_MR_d2931.nii.gz
      │   ├── OAS30137_MR_d3165.nii.gz
      │   ├── OAS30003_MR_d1631task.nii.gz
      │   ├── ...

# Usage for Segmentation with [nnU-Net](https://github.com/MIC-DKFZ/nnUNet.git)

The "task" marker is used to differentiate between data for training and validation. This marker can also be used for data splitting in [nnU-Net](https://github.com/MIC-DKFZ/nnUNet.git). Specifically, in **nnunet.training.net work_training.nnUNetTrainerV2.do_split()**.






