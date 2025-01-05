# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 21:14:20 2021

@author: 73239
"""
import torch.nn.functional as F
import torch
import multiprocessing as mp
from multiprocessing.context import Process
import numpy as np
from scipy.ndimage import gaussian_filter
import random
import os
import SimpleITK as sitk
import sys, time
import nibabel as nib
from scipy import ndimage
from tqdm import tqdm
import argparse
import os

def get_distance(f,spacing):
    """Return the signed distance."""

    dist_func = ndimage.distance_transform_edt
    distance = np.where(f, -(dist_func(f,sampling=spacing)),
                        dist_func(1-f,sampling=spacing))

    return distance


def get_head(img_path):
    
    temp = sitk.ReadImage(img_path)
    spacing = temp.GetSpacing()
    direction = temp.GetDirection()
    origin = temp.GetOrigin()
    
    return spacing,direction,origin

def copy_head_and_right_xyz(data,spacing,direction,origin):
    
    TrainData_new = data.astype('float32')
    TrainData_new = TrainData_new.transpose(2,1,0)
    TrainData_new = sitk.GetImageFromArray(TrainData_new)
    TrainData_new.SetSpacing(spacing)
    TrainData_new.SetOrigin(origin)
    TrainData_new.SetDirection(direction)
    
    return TrainData_new

def normalization(np_array):
    mean = np.mean(np_array)
    std = np.std(np_array)
    normed_array = (np_array-mean)/(std +1e-8)
    return normed_array,mean,std


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize() 
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int32) 
    resampler.SetReferenceImage(itkimage)  
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage) 
    return itkimgResampled

def get_mean_and_std(array,num):
    if num ==0:
        return 0,1
    mean = np.sum(array)/num
    temp = np.sum((array!=0)*(array-mean)*(array-mean))/num
    std = temp**0.5
    return mean,std

    
def ROI_normalization(ROI,mean,std):
    if np.sum(ROI!=0) != 0:
        self_mean,self_std = get_mean_and_std(ROI,np.sum(ROI!=0))
        once_normalized = (ROI!=0)*(ROI-self_mean)/self_std
        twice_normalized_ROI = (once_normalized*std + mean)*(ROI!=0)
    else:
        twice_normalized_ROI = np.zeros_like(ROI)
    return twice_normalized_ROI

'''=========================
CarveMix
============================'''
def Carve(Xi, Xj, Mi, Yi, Yj):
    
    brainmask = Xj!=0
    X = (Mi*Xi + (1 - Mi)*Xj)*brainmask 
        
    Y = Yi*brainmask
    
    return X,Y

def CarveMix(target_b, target_a, label_b, label_a,spacing):
    
    label = np.copy(label_b)
    # dis_array = get_distance(label>0,spacing)    #creat signed distance
    # c = c = random.uniform(0,1)#np.random.beta(1, 1)#[0,1]             #creat distance
    # c = 2*(c-0.5) #[-1.1]

    # if c>0:
    #     lam=c*np.min(dis_array)/2              #λl = -1/2|min(dis_array)|
    # else:
    #     lam=c*np.min(dis_array)               
    mask = label   #creat M 
    X,Y =  Carve(target_b, target_a, label_b, label_b, label_a)    #Patient_img,helthy_img, Patient_lab, 0
    return X,Y

'''=========================
Copy-Paste
=======
Data_shape: (m,h,w,l)
Label_shape: (h,w,l)
============================'''
def img_add(img_src, img_main, mask_src, mask_main):

    m, h, w, l = img_main.shape
    mask = np.asarray(mask_src)
    img_src,mask_02 = resize(img_src, mask_src,(h,w,l))
    paste_area = img_src * (mask_02>0)
    
    add_mask_main = mask_02 +(mask_02==0)*mask_main
    add_mask_main = add_mask_main.astype('float32')
    add_img_main = img_main*(mask_02==0) +paste_area
    return add_img_main, add_mask_main


     

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize() 
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int32) 
    resampler.SetReferenceImage(itkimage)  
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage) 
    return itkimgResampled

def random_flip_horizontal(Y,X,p=0.5):
    if random.random() < p:
        X = X[:,::-1,:,:]
        Y = Y[::-1,:,:]
    return Y,X

    


def Large_Scale_Jittering(mask, img, min_scale=0.1, max_scale=2.0):
    rescale_ratio = round(random.uniform(min_scale, max_scale),2)
    
    m, h, w, l = img.shape

    # rescale
    
    #img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    img,mask = resize(img,mask,(int(h*rescale_ratio),int(w*rescale_ratio),int(l*rescale_ratio)))#ndimage.interpolation.zoom(img,(1,rescale_ratio,rescale_ratio,rescale_ratio),mode='nearest')
    
    h_new, w_new, l_new = mask.shape
    #mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
    # mask = mask.resize((w_new, h_new), Image.NEAREST)

    # crop or padding
    x, y, z = int(random.uniform(0, abs(h_new - h))), int(random.uniform(0, abs(w_new - w))), int(random.uniform(0, abs(l_new - l)))
    if rescale_ratio <= 1.0 :   # padding
        img_pad = np.ones((m,h,w,l)) * 0
        mask_pad = np.zeros((h,w,l))
        img_pad[:,x:x+h_new,y:y+w_new,z:z+l_new] = img
        mask_pad[x:x+h_new,y:y+w_new,z:z+l_new] = mask
        mask_pad = np.round(mask_pad)
        return mask_pad, img_pad
    else:  # crop
        img_crop = img[:,x:x+h_new,y:y+w_new,z:z+l_new]
        #mask_crop = np.zeros(img_crop.shape[1::],dtype=np.uint8)
        mask_crop = mask[x:x+h_new,y:y+w_new,z:z+l_new]

        return mask_crop, img_crop


def resize(data,mask,size):
    """resize high-dimension tensor

    Args:
        data (`torch.Tensor`): tensor (C,d1,d2,d3)
        mask (`torch.Tensor`): tensor (d1,d2,d3)
        size (`Tuple` or `List`): target shape (D1,D2,D3)
        method (`str`): ['linear','nearest']
    """
    data1 = torch.from_numpy(data.copy()) 
    mask1 = torch.from_numpy(mask.copy()) 
    Data = F.interpolate(data1[None,...],size=size,mode='trilinear',align_corners=False).numpy()
    Mask = F.interpolate(mask1[None,None,...],size=size,mode='nearest').numpy()
    return Data[0,...], Mask[0,0,...]  # (C,D1,D2,D3), (D1,D2,D3)


"""
==========================================
The input must be nii.gz which contains 
import header information such as spacing.
Spacing will affect the generation of the
signed distance.
=========================================
"""
def generate_new_sample(image_a,image_b,label_b,local_count,opt,prefix,heterogeneous=True):
    spacing,direction,origin = get_head(image_a[0])
    target_a = []
    target_b = []



    flag = 0
    
    if heterogeneous:
        Mean = []
        Std = []
        temp_resize_nii_path = os.path.join(TEMPDir,'temp_image_%s.nii.gz'%str(local_count))
        temp_resize_nii_label_path = os.path.join(TEMPDir,'temp_label_%s.nii.gz'%str(local_count))#'/vol/biomedic3/xz2223/Project/nnUNetV1/nnUNet_raw/nnUNet_raw_data/Task006_SyntheTumour_90+10/TEMP/temp_label_%s.nii.gz'%str(local_count)
        #os.makedirs('/vol/biomedic3/xz2223/Project/nnUNetV1/nnUNet_raw/nnUNet_raw_data/Task006_SyntheTumour_90+10/TEMP',exist_ok=True)
        label1 = sitk.ReadImage(label_b[0])
        for i in range(opt.modality_num):
            data2 = nib.load(image_a[i]).get_fdata()   #mixed cases
            data1 = sitk.ReadImage(image_b[i])         #carved cases
            
            
            mod_i_image_path = image_b[i]
            
            if nib.load(image_a[i]).get_fdata().shape != nib.load(image_b[i]).get_fdata().shape:
                new_data1 = resize_image_itk(data1, data2.shape, resamplemethod=sitk.sitkLinear)
                sitk.WriteImage(new_data1,temp_resize_nii_path)
                mod_i_image_path = temp_resize_nii_path
                label_b = temp_resize_nii_label_path
                flag =1
        
        
            if flag == 1:
                new_label1 = resize_image_itk(label1, data2.shape, resamplemethod=sitk.sitkNearestNeighbor)
                sitk.WriteImage(new_label1,label_b)
                
            
            
            img = nib.load(image_a[i]).get_fdata()
            normed_array,mean,std = normalization(img)
            target_a.append(normed_array)
            Mean.append(mean)
            Std.append(std)
            img = nib.load(mod_i_image_path).get_fdata()
            normed_array,mean,std = normalization(img)
            target_b.append(normed_array)

    else:
        for i in range(opt.modality_num):
            target_a.append(nib.load(image_a[i]).get_fdata())
            target_b.append(nib.load(image_b[i]).get_fdata())



    #label_a = nib.load(label_a).get_fdata()
    # print(label_b)
    if isinstance(label_b, list):
        label_b = label_b[0]
    label_b = nib.load(label_b).get_fdata()
    label_a = np.zeros_like(label_b)
    
    target_a = np.array(target_a)
    target_b = np.array(target_b)
    
    
    
    new_target,new_label = CarveMix(target_b, target_a,label_b, label_a,spacing) 

    if heterogeneous:
        Mean = np.array(Mean)
        Std = np. array(Std)
        for i in range(opt.modality_num):
            new_target[i,:,:,:] = new_target[i,:,:,:] * Std[i] + Mean[i]
        
    if len(new_target.shape)<4:
        new_target.reshape(1,new_target.shape[0],new_target.shape[1],new_target.shape[2])
    target = []
    for i in range(opt.modality_num):
        target.append(copy_head_and_right_xyz(new_target[i],spacing,direction,origin))
    
    new_label = copy_head_and_right_xyz(new_label[0],spacing,direction,origin)
    # mask = copy_head_and_right_xyz(mask,spacing,direction,origin)
    
    s = str(local_count)
    
    for j in range(0,opt.modality_num):
        sitk.WriteImage(target[j], os.path.join(opt.imagesTr_path+'Mix', prefix +'_CarveMix_' + s + '_000%s.nii.gz'%str(j)))
    sitk.WriteImage(new_label, os.path.join(opt.labelsTr_path+'Mix', prefix +'_CarveMix_' + s + '.nii.gz'))
    # if i%1==0:
    #     sitk.WriteImage(mask, os.path.join(opt.mask_check_path, 'mask'+ '_CarveMix'+ s + '.nii.gz'))
        
    # return c
  
 
def generate_mulit_process(count,lock,Num,opt,prefix,X,PN=1):


    while count.value <Num:
        with lock:
            local_count = count.value
            count.value +=1
        image_a = []
        image_b = []
        rand_index_a = X[0][local_count] #Fake
        rand_index_b = X[1][local_count] #Health
        
        image_b = [os.path.join(opt.imagesTr_path, rand_index_a)]
        image_a = [os.path.join(opt.healthyimgTr, rand_index_b)]  #Health
        label_b = [os.path.join(opt.labelsTr_path, rand_index_a.replace('_0000.nii.gz','.nii.gz'))]
        #label_b = os.path.join(opt.labelsTr_path, Cases[rand_index_b] + '.nii.gz')
        #label_b_FPFN = os.path.join(opt.FPFN_path, Cases[rand_index_b] + '.nii.gz')
        generate_new_sample(image_a,image_b,label_b,local_count,opt,prefix)
        print('\r' + 'Completed: %.2f %%'%((local_count+1)/Num*100), end='', flush=True)
        #print('Process%s Creat%s'%(str(PN),str(local_count)))
        #bar.set_description(f'Process')
        s = str(local_count)
        csv_string = s + ',' + rand_index_a + ',' + rand_index_b.split('/')[0] + '\n'
        with open(opt.mixid_csv_path,'a') as f:
            f.write(csv_string)      
    
    

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
    parse = argparse.ArgumentParser()
    parse.add_argument("--generate_number","-num",default=200,type=int,
                       help="number of samples you want to creat: ")

    parse.add_argument("--mask_check_path","-mask",default="mask",type=str,
                       help="Path to save masks generated by CarveMix")
    parse.add_argument("--mixid_csv_path","-csv",default="/vol/biomedic3/xz2223/Project/nnUNetV1/nnUNet_raw/nnUNet_raw_data/Task006_SyntheTumour_90+10/CarveMixID.csv",type=str,
                       help="Path to save csv file")
    parse.add_argument("--modality_num","-mnum",default=1,type=int,
                       help='Modality Number')

    parse.add_argument("--imagesTr_path","-imgTr",default="/vol/biomedic3/xz2223/Project/nnUNetV1/nnUNet_raw/nnUNet_raw_data/Task006_SyntheTumour_90+10/imagesTr",type=str,
                       help="Path to raw tumour images with nnU-Net like data structure")
    parse.add_argument("--labelsTr_path","-labelTr",default="/vol/biomedic3/xz2223/Project/nnUNetV1/nnUNet_raw/nnUNet_raw_data/Task006_SyntheTumour_90+10/labelsTr",type=str,
                       help="Path to raw pseudo-labels generated by the junior model with nnU-Net like data structure")

    parse.add_argument("--healthyimgTr","-healthyimgTr",default="/vol/biomedic3/xz2223/Project/nnUNetV1/nnUNet_raw/nnUNet_raw_data/TaskXXX_SynthTumour/OASIS-200health-imagesTr",type=str,
                       help="Path to raw healthy images, like tumour-free images")


    
    opt = parse.parse_args()

    if not os.path.exists(opt.mask_check_path):
        os.makedirs(opt.mask_check_path,exist_ok=True)
        
    Mix_imagesTr = opt.imagesTr_path+'Mix'
    Mix_lablesTr = opt.imagesTr_path+'Mix'
    TEMPDir = opt.imagesTr_path+'TEMP'
    
    os.makedirs(Mix_imagesTr,exist_ok=True)
    os.makedirs(Mix_lablesTr,exist_ok=True)
    os.makedirs(TEMPDir,exist_ok=True)
    # os.makedirs(opt.labelsTr_path+'Mix',exist_ok=True)
    
    os.makedirs(opt.imagesTr_path,exist_ok=True)
    os.makedirs(opt.labelsTr_path,exist_ok=True)

    
    FCases = os.listdir(opt.imagesTr_path)
    # simpleCases = [case.split('.')[0] for i, case in enumerate(Cases) if 'Mix' not in case]
    # Cases = simpleCases
    prefix = FCases[0].split('_')[0]

    
    Num = opt.generate_number #5#len(FCases)
    #num = len(Cases)
    print('all_set_size: ',Num)
    FCases.sort()
    HCases = os.listdir(opt.healthyimgTr)
    
    
    


    print(opt)
    
    X1ID = random.sample(FCases, len(FCases)) + random.sample(FCases, len(FCases))
    X2ID = random.sample(HCases, len(HCases))#[HCases[random.randint(0,len(HCases)-1)] for i in 2*len(FCases)]
    # X2ID = [os.path.join(i,'/flair.nii.gz') for i in X2ID]
    X = [X1ID,X2ID]    
    
    with open(opt.mixid_csv_path,'w') as f:
        f.write('id,id1,id2\n') 
    
    """
    Start generating augmentated samples
    """
    process_lock = mp.Lock()
    count = mp.Value('i',0)

    
    #generate_mulit_process(count,process_lock,Num,opt,prefix,X)
    proc_list = [mp.Process(target = generate_mulit_process,args=((count,process_lock,Num,opt,prefix,X,i))) for i in range(10)]
    [p.start() for p in proc_list]
    [p.join() for p in proc_list]
    
    
