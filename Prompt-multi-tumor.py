import os
#import shutil
import random
import numpy as np
import multiprocessing as mp
from multiprocessing.context import Process

def resect_tumor_command(input_img_path,output_img_path,output_lab_path,task):
    cmd = 'python resect-multi-tumor.py %s %s %s -task %s'%(input_img_path,output_img_path,output_lab_path,task)

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
    Dir_path = 'DemoData'
    Case = os.listdir(Dir_path)

    nnUNet_Task_Dir = 'nnUNet_DataFramework/Task100_OASIS-PLBTS'
    print('Clearn Task Dir......')
    os.system('rm -rf %s/*'%nnUNet_Task_Dir)
    imagesTr = os.path.join(nnUNet_Task_Dir,'imagesTr')
    labelsTr = os.path.join(nnUNet_Task_Dir,'labelsTr')
    os.makedirs(imagesTr,exist_ok=True)
    os.makedirs(labelsTr,exist_ok=True)

    # exit()
    print('Start-Prompt......')
    CMD = []
    random.seed(527)
    Case = random.sample(Case,len(Case))
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
            output_img_path = os.path.join(imagesTr,case+'val_0000.nii.gz')
            output_lab_path = os.path.join(labelsTr,case+'val.nii.gz')
            #cmd = [input_img_path,output_img_path,output_lab_path,task]
        CMD.append(resect_tumor_command(input_img_path,output_img_path,output_lab_path,task))

    """
    Start generating augmentated samples
    """

    process_lock = mp.Lock()
    count = mp.Value('i',0)
    Num = len(CMD)
    
    # generate_mulit_process(count, process_lock,Num,CMD,i=1)
    proc_list = [mp.Process(target = generate_mulit_process,args=((count, process_lock,Num,CMD,i))) for i in range(30)]
    [p.start() for p in proc_list]
    [p.join() for p in proc_list]
