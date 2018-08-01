import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from PIL import Image
plt.switch_backend('agg')

def load_dicom_volume(patient_dir):
    '''
    Load a dicom series into a 3D numpy volume
    Args:
        patient_dir:
    '''
    reader = sitk.GetGDCMSeriesFileNames(patient_dir)
    reader.SetFileNames(patient_dir)

    images = reader.Execute()
    size = images.GetSize()

    print( "Image size:", size[0], size[1], size[2] )

    np_array = sitk.GetArrayFromImage(images)

    return np_array

def create_dataframe(patient_dir):
    '''
    Create a pandas dataframe containing directories to each modality
    '''
    patient_id = os.listdir(patient_dir)  
    t1_lis = []
    t2_lis = []
    t1C_lis = []
    flair_lis = []
    pathology_lis = []
    for id_ in patient_id:
        id_path = os.path.join(patient_dir,id_)
        img_lis = os.listdir(id_path)
        
        if 'FLAIR.nii' not in img_lis:
            flair_lis.append('NA')
        for img in img_lis:
            
            print(img)
            if img == 'T1.nii':
                t1_lis.append(os.path.join(id_path,img))
            elif img == 'T2.nii':
                t2_lis.append(os.path.join(id_path,img))
            elif img =='T1C.nii':
                t1C_lis.append(os.path.join(id_path,img))
            elif img == 'FLAIR.nii':
                flair_lis.append(os.path.join(id_path,img))
            elif img == 'Pathology.PNG':
                pathology_lis.append(os.path.join(id_path,img))
                



    dataframe = pd.DataFrame({'patient_ID':patient_id,'T1':t1_lis,'T2':t2_lis,'T1C':t1C_lis,'FLAIR':flair_lis,'pathology':pathology_lis})

    return dataframe


def load_volume(filename):
    '''
    Load a nifti image and store as numpy array, resize according to different modality
    Args:
        filename: full path to the image (string)
    Outputs:
        np_array: dimension z,x,y
    '''
    print(filename)

    volume = sitk.ReadImage(filename)
    np_array = sitk.GetArrayFromImage(volume)
    np_array = np.resize(np_array,[128,256,256])
    print(np_array.shape)
    return np_array

def load_image(filename):
    '''
    Load and resize a 2D image
    '''
    print(filename)
    img = Image.open(filename)
    print(img.size)
    img = img.resize((1024,1024),Image.NEAREST)
    print(img.size)
    img = np.array(img)

    img = img.astype(np.float)
    print(img.shape)
    plt.title('Original Image: ')
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(os.path.join('./results','Original'))
    plt.close('all')
    print(img)
    img = np.transpose(img,[2,0,1])
    img = np.resize(img,(3,1024,1024))
    print(img.shape)
    
    
    return img


def plot_reconstruction(original_img, recon_img,iters,epoch,save_dir):
    '''
    '''
    title = 'Original_Recon_'+str(iters)+'_iter_'+str(epoch)+'_epoch'
    
    # print(original_img.shape)
    # print(recon_img.shape)

    original_img = np.transpose(original_img,[2,1,0])
    recon_img = np.transpose(recon_img,[2,1,0])
    print(original_img.shape)

    fig = plt.figure(figsize=(10,10))
    plt.subplot(211)
    plt.title('Reconstructed Image: ')
    plt.imshow(recon_img)
    plt.axis('off')
    

    plt.subplot(212)
    
    plt.imshow(original_img)
    plt.axis('off')
    plt.title('Original Image: ')
    plt.savefig(os.path.join(save_dir,title))
    plt.close('all')