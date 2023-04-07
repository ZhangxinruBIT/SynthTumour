# PL-BTS: Unsupervised Brain Tumor Segmentation with Image-based Prompts

Please also cite this paper if you are using PL-BTS for your research! For more information about PL-BTS, please read the following [papper](https://arxiv.org/abs/2304.01472)

     @misc{zhang2023unsupervised,
           title={Unsupervised Brain Tumor Segmentation with Image-based Prompts}, 
           author={Xinru Zhang and Ni Ou and Chenghao Liu and Zhizheng Zhuo and Yaou Liu and Chuyang Ye},
           year={2023},
           eprint={2304.01472},
           archivePrefix={arXiv},
           primaryClass={cs.CV}
     }


Figure 1 provides an overview of the method proposed in this study.
![image](Figs/overview.png)


Our work drew inspiration from the amazing fepegar's [resector](https://github.com/fepegar/resector) tool, which served as the main basis for generating the 3D tumor-like shape. We made certain revisions to the tool to suit our specific needs.
# Installation
```
conda create --name PLBTS python=3.7.7
conda activate PLBTS
pip install light-the-torch
ltt install torch
pip install git+https://github.com/fepegar/resector
git clone https://github.com/ZhangxinruBIT/PL-BTS.git
```

# Usage for Generation
**Run demo**
```
cd PL-BTS
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
cd PL-BTS
python Prompt-multi-tumor.py

```

After that we could get the generated folds as below:

     .
     └── Task100_OASIS-PLBTS
         ├── imagesTr
         │   ├── OAS30003_MR_d1631_0000.nii.gz
         │   ├── OAS30135_MR_d2931val_0000.nii.gz
         │   └── OAS30137_MR_d3165_0000.nii.gz
         └── labelsTr
             ├── OAS30003_MR_d1631.nii.gz
             ├── OAS30135_MR_d2931val.nii.gz
             └── OAS30137_MR_d3165.nii.gz

# Usage for Segmentation with [nnU-Net](https://github.com/MIC-DKFZ/nnUNet.git)

The "val" marker is used to differentiate between data for training and validation. This marker can also be used for data splitting in [nnU-Net](https://github.com/MIC-DKFZ/nnUNet.git). Specifically, in **nnunet.training.net work_training.nnUNetTrainerV2.do_split()**.

We did not focus on imitating the tumor texture, as establishing the accurate tumor texture from MRI images is challenging. Instead, we proposed a validation task to prevent the model from overfitting, ensuring better results.

Figure 2 illustrates the importance of the validation task in preventing overfitting caused by manual texture on MRI. It presents a schematic of the training process of the prompt model. Without the validation task, model selection may occur after the turning point of test accuracy, leading to overfitting, even if a validation set comprising synthetic data is separated from the training set. However, with the inclusion of the validation task, the turning point can be better identified for effective model selection.
<div align=center><img width="650" height="450" src="Figs/loss.png"/></div>






