"""Console script for resector."""
from lib2to3.pgen2.token import PLUS
import sys
import time
import click
from pathlib import Path
import os
from scipy.ndimage import gaussian_filter
import nibabel as nib
import numpy as np
from scipy import ndimage
import random


@click.command()
@click.argument('input-path', type=click.Path(exists=True))
#@click.argument('parcellation-path', type=click.Path(exists=True))
@click.argument('output-image-path', type=click.Path())
@click.argument('output-label-path', type=click.Path())
@click.option('--seed', '-s', type=int)
@click.option('--min-volume', '-miv', type=int, default=20000, show_default=True) #V5:10000  #V6 20000  #V7 30000
@click.option('--max-volume', '-mav', type=int, default=80000, show_default=True) #V5:70000  #V6 80000
@click.option('--volumes-path', '-p', type=click.Path(exists=True))
@click.option('--simplex-path', '-n', type=click.Path(exists=True))
@click.option('--std-blur', type=float)
@click.option('--shape', type=click.Choice(['ellipsoid', 'cuboid', 'noisy']), default='noisy', show_default=True)
@click.option('--texture', type=click.Choice(['dark', 'random', 'csf']), default='dark', show_default=True)
@click.option('--center-ras', '-r', nargs=3, type=float)
@click.option('--wm-lesion/--no-wm-lesion', '-w', type=bool, default=False, show_default=True)
@click.option('--clot/--no-clot', '-b', type=bool, default=False, show_default=True)
@click.option('--verbose/--no-verbose', '-v', type=bool, default=False, show_default=True)
@click.option('--debug-dir', '-d', type=click.Path(file_okay=False))
@click.option('--cleanup/--no-cleanup', '-c', default=False, show_default=True)
@click.option('--multi-lesion/--no-multi-lesion', '-ml', default=False, show_default=True)
@click.option('-task', type = bool, default=False, show_default=True)
#@click.option('--high-prompt-percent', '-hper', type=float, default=1.0, show_default=True)
def main(
        input_path,
        #parcellation_path,
        output_image_path,
        output_label_path,
        seed,
        min_volume,
        max_volume,
        volumes_path,
        simplex_path,
        std_blur,
        shape,
        texture,
        center_ras,
        wm_lesion,
        clot,
        verbose,
        debug_dir,
        cleanup,
        multi_lesion,
        task,
        #high_propmt_percent,
        
        ):
    import torchio as tio
    import resector

    if seed is not None:
        import torch
        torch.manual_seed(seed)

    if debug_dir is not None:
        resector.io.debug_dir = Path(debug_dir).expanduser().absolute()

    brain_mask_path, existed = ensure_images(
        input_path,
       # parcellation_path,
    )

    try:
        if volumes_path is not None:
            import pandas as pd
            df = pd.read_csv(volumes_path)
            volumes = df.Volume.values
            kwargs = dict(volumes=volumes)
        else:
            kwargs = dict(volumes_range=(min_volume, max_volume))
        if std_blur is not None:
            kwargs['sigmas_range'] = std_blur, std_blur
        kwargs['simplex_path'] = simplex_path
        kwargs['wm_lesion_p'] = wm_lesion
        kwargs['clot_p'] = clot
        kwargs['verbose'] = verbose
        kwargs['shape'] = shape
        kwargs['texture'] = texture
        kwargs['center_ras'] = center_ras
        import nibabel as nib
        if not multi_lesion:       
            transform = tio.Compose((
                tio.ToCanonical(),
                resector.RandomResection(**kwargs),
            ))
            ###########################################################
            subject = tio.Subject(
                image=tio.ScalarImage(input_path),
                resection_resectable_left=tio.LabelMap(brain_mask_path),
                resection_resectable_right=tio.LabelMap(brain_mask_path),
                resection_gray_matter_left=tio.LabelMap(brain_mask_path),
                resection_gray_matter_right=tio.LabelMap(brain_mask_path),
                resection_noise=tio.ScalarImage(brain_mask_path),
            )
            with resector.timer('RandomResection', verbose):
                transformed = transform(subject)
            with resector.timer('Saving images', verbose):
                # transformed['image'].save(output_image_path)
                transformed['label'].save(output_label_path)
            lab1 = nib.load(output_label_path).get_fdata()
            dt = nib.load(input_path)
            nib.Nifti1Image(lab1,affine=dt.affine,header=dt.header).to_filename(output_label_path)
            ###########################################################
            subject = tio.Subject(
                image=tio.ScalarImage(input_path),
                resection_resectable_left=tio.LabelMap(output_label_path),
                resection_resectable_right=tio.LabelMap(output_label_path),
                resection_gray_matter_left=tio.LabelMap(output_label_path),
                resection_gray_matter_right=tio.LabelMap(output_label_path),
                resection_noise=tio.ScalarImage(output_label_path),
            )
            with resector.timer('RandomResection', verbose):
                transformed = transform(subject)
            with resector.timer('Saving images', verbose):
                # transformed['image'].save(output_image_path)
                transformed['label'].save(output_label_path)  
            lab2 = nib.load(output_label_path).get_fdata()        
            ###########################################################
            Lab = lab1+lab2 #nib.load(output_label_path).get_fdata()
            #exit()
            return_code = 0   
            
        dt = nib.load(input_path)
        Img = nib.load(input_path).get_fdata()
        # nib.Nifti1Image(Img,affine=dt.affine).to_filename(output_image_path)
        # exit()
        prompt_img,prompt_lab = prompt(Img,Lab,high_propmt_percent = 0.9,task = task)
        nib.Nifti1Image(prompt_lab.astype('float32'),affine=dt.affine).to_filename(output_label_path)
        nib.Nifti1Image(prompt_img,affine=dt.affine).to_filename(output_image_path)
            
    except Exception as e:
        return_code = 1
        raise
    finally:
        if not existed and cleanup:
            with resector.timer('Cleaning up', verbose):
                for p in brain_mask_path: p.unlink()
                for p in brain_mask_path: p.unlink()
                brain_mask_path.unlink()

    return return_code


def ensure_images(input_path):
    import resector
    input_path = Path(input_path)
    output_dir = input_path.parent
    stem = input_path.name.split('.')[0]
    brain_mask_path = os.path.join(output_dir / f'{stem}_brainmask.nii.gz')
    assert os.path.exists(brain_mask_path),'lack of brain mask %s to debind the lesion area, using bet to get it'%brain_mask_path
    existed = True    
    return brain_mask_path,existed


        
#216,217,218,219,230
def prompt(img,lab,high_propmt_percent,task=False):
    import random
    timg = np.copy(img)
    tlab = np.copy(lab)
 
    test = gaussian_filter(timg,sigma=5) #v256

    if random.random()<high_propmt_percent:
        t = np.random.uniform(1.5,5)
        sub_mask = tlab>0
        sub_mask = sub_mask.astype('float32') 
        sub_mask = gaussian_filter(sub_mask,1.0)      
        if not task:
            new_img = img*(1-sub_mask) + test*t*(sub_mask)
            mask = sub_mask>0.1
            
        else:
            mask = np.copy(lab)
            mask = mask>0
            mask = mask.astype('float32')
            dis = get_distance(mask)
            dis = dis/np.max(dis)
            if random.random()>1.0:
                new_img = img*(mask==0) + img*t*(mask!=0)*dis + img*(mask!=0)*(1-dis)
            else:
                new_img = img*(mask==0) + img*t*(mask!=0)
        
    else:
        sub_mask1 = tlab>0
        sub_mask1 = sub_mask1.astype('float32') 
        sub_mask1 = gaussian_filter(sub_mask1,1.0)   
        t1 = np.random.uniform(1.5,5)
        
        sub_mask2 = tlab>1
        sub_mask2 = sub_mask2.astype('float32') 
        sub_mask2 = gaussian_filter(sub_mask2,1.0)   
        t2 = np.random.uniform(0.8,1.2)
        if not task:
            new_img = img*(1-sub_mask1) + test*t1*(sub_mask1)
            new_img = new_img*(1-sub_mask2) + test*t2*(sub_mask2)
            mask = sub_mask1>0.1
            
        else:
            t2 = np.random.uniform(0.8,1.2)
            mask = np.copy(lab)
            mask1 = mask>0
            mask1 = mask1.astype('float32')
            
            mask2 = mask>1
            mask2 = mask2.astype('float32')
            mask = mask>0
            mask = mask.astype('float32')

            if random.random()>1.0:
                dis = get_distance(mask1)
                dis = dis/np.max(dis)
                new_img = img*(mask1==0) + img*t1*(mask1!=0)*dis + img*(mask1!=0)*(1-dis)
                dis = get_distance(mask2)
                dis = dis/np.max(dis)
                new_img = new_img*(mask2==0) + img*t2*(mask2!=0)*dis + img*(mask2!=0)*(1-dis)
            else:
                new_img = img*(mask1==0) + img*t1*(mask1!=0) 
                new_img = new_img*(mask2==0) + img*t2*(mask2!=0)       
        

    return new_img,mask





def get_distance(f,spacing=[1,1,1]):
    """Return the signed distance."""

    dist_func = ndimage.distance_transform_edt
    distance = np.where(f, dist_func(f,sampling=spacing),-dist_func(1-f,sampling=spacing))

    return distance       
    


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    sys.exit(main())  # pragma: no cover
