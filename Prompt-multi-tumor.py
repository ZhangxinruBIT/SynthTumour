import os
#import shutil
import random
import numpy as np
import multiprocessing as mp
from multiprocessing.context import Process

def resect_tumor_command(input_img_path,output_img_path,output_lab_path,task):
    cmd = 'python resect-multi-tumor.py %s %s %s -task %s'%(input_img_path,output_img_path,output_lab_path,task)
    #cmd = 'python resect3.py %s %s %s -miv  %s -mav %s -task %s'%(input_img_path,output_img_path,output_lab_path,str(5000),str(20000),task)
    return cmd
 
 

def generate_mulit_process(count,lock,Num,CMD,i):
    while count.value <Num:
        with lock:
            local_count = count.value
            count.value +=1
        cmd = CMD[local_count]
        os.system(cmd)
        a=1 




if __name__ == '__main__':
    exit()
    Dir_path = '/data7/xinruzhang/DATA/OASIS3/pair_regitstration_brainmask'
    Case = os.listdir(Dir_path)

    nnUNet_Task_Dir = '/data7/xinruzhang/nnUNet/nnUNet_raw/nnUNet_raw_data/Task270_OASIS-multi-0.9-t-0.8-1.2V6-train100-val100-valway1-GS5'
    print('=======Clearn Task Dir======')
    os.system('rm -rf %s/*'%nnUNet_Task_Dir)
    imagesTr = os.path.join(nnUNet_Task_Dir,'imagesTr')
    labelsTr = os.path.join(nnUNet_Task_Dir,'labelsTr')
    os.makedirs(imagesTr,exist_ok=True)
    os.makedirs(labelsTr,exist_ok=True)
    print('=======Start-COPY-REAL-Data======')
    Brats_lab_path = '/data6/xinruzhang/nnUNet/nnUNet_raw/nnUNet_raw_data/Task101_Brats_FLAIR_wholetumor_100train_baseline/labelsTs/*'
    Brats_img_path = '/data6/xinruzhang/nnUNet/nnUNet_raw/nnUNet_raw_data/Task101_Brats_FLAIR_wholetumor_100train_baseline/imagesTs/*'
    os.system('cp -r %s %s'%(Brats_lab_path,labelsTr))
    os.system('cp -r %s %s'%(Brats_img_path,imagesTr))
    # exit()
    print('=======Start-Prompt======')
    CMD = []
    random.seed(527)
    Case = random.sample(Case,200)
    Case_task = random.sample(Case,int(len(Case)*0.5))
    for case in Case:
        if case not in Case_task:
            case_path = os.path.join(Dir_path,case)
            input_img_path = os.path.join(case_path,'flair.nii.gz')
            output_img_path = os.path.join(imagesTr,case+'_0000.nii.gz')
            output_lab_path = os.path.join(labelsTr,case+'.nii.gz')
            task=False
        else:
            task = True
            case_path = os.path.join(Dir_path,case)
            input_img_path = os.path.join(case_path,'flair.nii.gz')
            output_img_path = os.path.join(imagesTr,case+'task_0000.nii.gz')
            output_lab_path = os.path.join(labelsTr,case+'task.nii.gz')
            #cmd = [input_img_path,output_img_path,output_lab_path,task]
        CMD.append(resect_tumor_command(input_img_path,output_img_path,output_lab_path,task))
    #CMD = CMD[0:8]  
    #print(CMD)
    """
    Start generating augmentated samples
    """
    #CMD = CMD[0:30]
    print(len(CMD))
    process_lock = mp.Lock()
    count = mp.Value('i',0)
    Num = len(CMD)
    
    # generate_mulit_process(count, process_lock,Num,CMD,i=1)
    proc_list = [mp.Process(target = generate_mulit_process,args=((count, process_lock,Num,CMD,i))) for i in range(30)]
    [p.start() for p in proc_list]
    [p.join() for p in proc_list]