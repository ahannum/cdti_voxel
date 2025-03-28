import sys
sys.path.append('Software/CarDpy-master')
import glob
import os
import scipy.io
from scipy.io import loadmat
import nrrd
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
import matplotlib.pyplot as plt
import time
from cDTIpy.Data_Sorting.Diffusion            import *
from skimage.measure import label
import numpy as np
from   cDTIpy.Diffusion.DTI import DTI_recon
from cDTIpy.Colormaps.Diffusion import cDTI_Colormaps_Generator

cDTI_cmaps = cDTI_Colormaps_Generator()



import time

def Header_Reader(header_path):
    with open(header_path) as f:
        lines = f.readlines()
    HeadersDict = dict()
    for idx in range(len(lines)):
        key_word              = lines[idx].split(':')[0]
        key_value             = lines[idx].split(':')[1]
        key_value             = key_value.strip()
        key_value_clear       = key_value.split(' ')[0]
        HeadersDict[key_word] = key_value_clear
    return HeadersDict


def calc_95_CI(data, number_of_bins = 1000):
    import numpy as np
#     min_value = np.min(data)
#     max_value = np.max(data)
#     bin_width = (max_bootstrap - min_bootstrap) / 100
    
    [vals, bin_edges] = np.histogram(data, bins = number_of_bins)
    h = vals / np.sum(vals)  
    ### Find lower bound index (j)
    j = 0
    test = 0
    while (test <= 0.025):
        test = test + h[j]
        if (test <= 0.025):
            j = j + 1
        if (test > 0.025):
            if j == 0:
                j = j
            else:
                j = j - 1
    ### Find upper bound index (k)
    k = len(h) - 1
    test = 0
    while (test <= 0.025):
        test = test + h[k]
        if (test <= 0.025):
            k = k - 1
        if (test > 0.025):
            k = k + 1

    lower_bound = ((bin_edges[j + 1] - bin_edges[j]) /2) + bin_edges[j]
    upper_bound = ((bin_edges[k - 1] - bin_edges[k]) /2) + bin_edges[k]
    return [lower_bound, upper_bound]

def Invariant_Uncertainty(Bootstrapped_Standard_DTI_Metrics):
    import numpy as np
    
    rows = Bootstrapped_Standard_DTI_Metrics['MD'].shape[0]
    cols = Bootstrapped_Standard_DTI_Metrics['MD'].shape[1]
    slcs = Bootstrapped_Standard_DTI_Metrics['MD'].shape[2]
    
    dMD = np.zeros([rows, cols, slcs])
    dTR = np.zeros([rows, cols, slcs])
    dFA = np.zeros([rows, cols, slcs])
    dMO = np.zeros([rows, cols, slcs])
    dAD = np.zeros([rows, cols, slcs])
    dRD = np.zeros([rows, cols, slcs])

    mMD = np.zeros([rows, cols, slcs])
    mTR = np.zeros([rows, cols, slcs])
    mFA = np.zeros([rows, cols, slcs])
    mMO = np.zeros([rows, cols, slcs])
    mAD = np.zeros([rows, cols, slcs])
    mRD = np.zeros([rows, cols, slcs])
    for x in range(rows):
        for y in range(cols):
            for slc in range(slcs):
                data   = Bootstrapped_Standard_DTI_Metrics['MD'][x, y, slc, :].flatten()
                data_2 = data[~(np.isnan(data))]
#                 [LB, UB] = calc_95_CI(data_2)
                data_sorted       = np.sort(data_2)
                Upper_Bound_Index = len(data_sorted) - int(np.round(0.025 * len(data_sorted), 0)) - 1
                Lower_Bound_Index = int(np.round(0.025 * len(data_sorted), 0)) - 1
                UB = data_sorted[Upper_Bound_Index]
                LB = data_sorted[Lower_Bound_Index]
                dMD[x, y, slc] = UB - LB
                mMD[x, y, slc] = np.median(data_2)
                del data, data_2, data_sorted, Upper_Bound_Index, Lower_Bound_Index, LB, UB

                data = Bootstrapped_Standard_DTI_Metrics['TR'][x, y, slc, :].flatten()
                data_2 = data[~(np.isnan(data))]
#                 [LB, UB] = calc_95_CI(data_2)
                data_sorted       = np.sort(data_2)
                Upper_Bound_Index = len(data_sorted) - int(np.round(0.025 * len(data_sorted), 0)) - 1
                Lower_Bound_Index = int(np.round(0.025 * len(data_sorted), 0)) - 1
                UB = data_sorted[Upper_Bound_Index]
                LB = data_sorted[Lower_Bound_Index]
                dTR[x, y, slc] = UB - LB
                mTR[x, y, slc] = np.median(data_2)
                del data, data_2, data_sorted, Upper_Bound_Index, Lower_Bound_Index, LB, UB

                data = Bootstrapped_Standard_DTI_Metrics['FA'][x, y, slc, :].flatten()
                data_2 = data[~(np.isnan(data))]
#                 [LB, UB] = calc_95_CI(data_2)
                data_sorted       = np.sort(data_2)
                Upper_Bound_Index = len(data_sorted) - int(np.round(0.025 * len(data_sorted), 0)) - 1
                Lower_Bound_Index = int(np.round(0.025 * len(data_sorted), 0)) - 1
                UB = data_sorted[Upper_Bound_Index]
                LB = data_sorted[Lower_Bound_Index]
                dFA[x, y, slc] = UB - LB
                mFA[x, y, slc] = np.median(data_2)
                del data, data_2, data_sorted, Upper_Bound_Index, Lower_Bound_Index, LB, UB
                
                data = Bootstrapped_Standard_DTI_Metrics['MO'][x, y, slc, :].flatten()
                data_2 = data[~(np.isnan(data))]
#                 [LB, UB] = calc_95_CI(data_2)
                if len(data_2) == 0:
                    data_sorted       = []
                    Upper_Bound_Index = []
                    Lower_Bound_Index = []
                    UB = np.nan
                    LB = np.nan
                else:
                    data_sorted       = np.sort(data_2)
                    Upper_Bound_Index = len(data_sorted) - int(np.round(0.025 * len(data_sorted), 0)) - 1
                    Lower_Bound_Index = int(np.round(0.025 * len(data_sorted), 0)) - 1
                    UB = data_sorted[Upper_Bound_Index]
                    LB = data_sorted[Lower_Bound_Index]
                dMO[x, y, slc] = UB - LB
                mMO[x, y, slc] = np.median(data_2)
                del data, data_2, data_sorted, Upper_Bound_Index, Lower_Bound_Index, LB, UB
                
                data = Bootstrapped_Standard_DTI_Metrics['AD'][x, y, slc, :].flatten()
                data_2 = data[~(np.isnan(data))]
#                 [LB, UB] = calc_95_CI(data_2)
                data_sorted       = np.sort(data_2)
                Upper_Bound_Index = len(data_sorted) - int(np.round(0.025 * len(data_sorted), 0)) - 1
                Lower_Bound_Index = int(np.round(0.025 * len(data_sorted), 0)) - 1
                UB = data_sorted[Upper_Bound_Index]
                LB = data_sorted[Lower_Bound_Index]
                dAD[x, y, slc] = UB - LB
                mAD[x, y, slc] = np.median(data_2)
                del data, data_2, data_sorted, Upper_Bound_Index, Lower_Bound_Index, LB, UB

                data = Bootstrapped_Standard_DTI_Metrics['RD'][x, y, slc, :].flatten()
                data_2 = data[~(np.isnan(data))]
#                 [LB, UB] = calc_95_CI(data_2)
                data_sorted       = np.sort(data_2)
                Upper_Bound_Index = len(data_sorted) - int(np.round(0.025 * len(data_sorted), 0)) - 1
                Lower_Bound_Index = int(np.round(0.025 * len(data_sorted), 0)) - 1
                UB = data_sorted[Upper_Bound_Index]
                LB = data_sorted[Lower_Bound_Index]
                dRD[x, y, slc] = UB - LB
                mRD[x, y, slc] = np.median(data_2)
                del data, data_2, data_sorted, Upper_Bound_Index, Lower_Bound_Index, LB, UB
                
    Invariant_Uncertainties = dict()
    Invariant_Uncertainties['dMD'] =  dMD
    Invariant_Uncertainties['dTR'] =  dTR
    Invariant_Uncertainties['dFA'] =  dFA
    Invariant_Uncertainties['dMO'] =  dMO
    Invariant_Uncertainties['dAD'] =  dAD
    Invariant_Uncertainties['dRD'] =  dRD
    Invariant_Medians = dict()
    Invariant_Medians['mMD'] =  mMD
    Invariant_Medians['mTR'] =  mTR
    Invariant_Medians['mFA'] =  mFA
    Invariant_Medians['mMO'] =  mMO
    Invariant_Medians['mAD'] =  mAD
    Invariant_Medians['mRD'] =  mRD
    return [Invariant_Uncertainties, Invariant_Medians]



overal_start_time = time.time()

#main_data_path         = '/Volumes/T7/Projects/01_cDTI_EPI_Distortion_Correction/New_Data/DTI_Bootstrapping'
#main_image_path        = '/Volumes/T7/Projects/01_cDTI_EPI_Distortion_Correction/New_Data/NifTis'
#main_segmentation_path = '/Volumes/T7/Projects/01_cDTI_EPI_Distortion_Correction/New_Data/Segmentations/'


main_data_path         = '/home/ahannum/Documents/Voxel/BootstrapUltron/'
main_image_path        = '/home/ahannum/Documents/Voxel/BootstrapUltron/'
main_segmentation_path = '/home/ahannum/Documents/Voxel/BootstrapUltron/'
volunteers        = ['V001','V002','V003','V004','V005','V006','V007','V008','V009','V010','V011',]

for subject_folder in volunteers:

    #DTI_folder  = 'cDTI'
    Folder       = ['vol_2.0res_3.0sl','vol_2.0res_5.0sl','vol_2.0res_8.0sl', 'vol_2.5res_3.0sl','vol_2.5res_5.0sl','vol_2.5res_8.0sl', 'vol_3.0res_3.5sl','vol_3.0res_5.5sl','vol_3.0res_8.0sl']
    # Folder      = 'DiVO_06_10'
    Data_Folder = '07_Bootstrapping'
    Evals_Folder = '07_Bootstrapping'


    for flid in range(len(Folder)):
        inpath_1 = glob.glob(os.path.join(main_segmentation_path, subject_folder,  Folder[flid],  '06_cDTI_Maps/Interpolated_mask.mat'))[0]
        print('Path to NRRD images:  ' + inpath_1)
        NRRD = loadmat(os.path.join(inpath_1))['Mask']

        inpath_2 = glob.glob(os.path.join(main_image_path, subject_folder,  Folder[flid], Data_Folder, 'Registered_1000_Bootstraps.header'))[0]
        print('Path to header:       ' + inpath_2)
        Header = Header_Reader(inpath_2)

        inpath_3 = glob.glob(os.path.join(main_image_path, subject_folder,  Folder[flid], Data_Folder, 'Registered_1000_Bootstraps.nii'))[0]
        print('Path to NifTi images: ' + inpath_3)
        NifTi, affine_matrix, NifTi_VoxRes = load_nifti(inpath_3, return_voxsize = True)

        Header['X Resolution'] = NifTi_VoxRes[0]
        Header['Y Resolution'] = NifTi_VoxRes[1]
        Header['Z Resolution'] = NifTi_VoxRes[2]
        print(Header)

        inpath_4 = glob.glob(os.path.join(main_data_path, subject_folder,  Folder[flid], Evals_Folder, 'Standard_DTI_Metrics_1000_Bootstraps.mat'))[0]
        print('Path to DTI Metrics:  ' + inpath_4)
        mat = scipy.io.loadmat(inpath_4)
        
        Bootstrapped_MD = mat['MD']
        Bootstrapped_TR = mat['TR']
        Bootstrapped_FA = mat['FA']
        Bootstrapped_MO = mat['MO']
        Bootstrapped_AD = mat['AD']
        Bootstrapped_RD = mat['RD']

        Bootstrapped_Standard_DTI_Metrics = dict()
        Bootstrapped_Standard_DTI_Metrics['MD'] = Bootstrapped_MD
        Bootstrapped_Standard_DTI_Metrics['TR'] = Bootstrapped_TR
        Bootstrapped_Standard_DTI_Metrics['FA'] = Bootstrapped_FA
        Bootstrapped_Standard_DTI_Metrics['MO'] = Bootstrapped_MO
        Bootstrapped_Standard_DTI_Metrics['AD'] = Bootstrapped_AD
        Bootstrapped_Standard_DTI_Metrics['RD'] = Bootstrapped_RD
        
        [Invariant_Uncertainties, Invariant_Medians] = Invariant_Uncertainty(Bootstrapped_Standard_DTI_Metrics)

        matrix = NifTi
        myocardial_mask_smoothed = np.copy(NRRD)
        myocardial_mask_smoothed_nan = np.copy(myocardial_mask_smoothed)
        myocardial_mask_smoothed_nan = myocardial_mask_smoothed_nan.astype('float')
        myocardial_mask_smoothed_nan [myocardial_mask_smoothed_nan == 0] = np.nan

        vmax_image = 200
        main_maps_outpath = '/home/ahannum/Documents/Voxel/BootstrapUltron/Uncertainty_Maps'
        if not os.path.exists(os.path.join(main_maps_outpath, subject_folder, Folder[flid])):
                # If it doesn't exist, create it
                os.makedirs(os.path.join(main_maps_outpath, subject_folder, Folder[flid]))

        for slc in range(myocardial_mask_smoothed_nan.shape[2]):

            print('Slice %i of %i' %(slc + 1, myocardial_mask_smoothed_nan.shape[2]))
            fig = plt.figure(figsize = (15, 15), dpi= 100)
            fig.patch.set_facecolor('white')
            fig.suptitle('Uncertainty Bootstrap cDTI Maps for Slice %i' %(slc + 1), fontsize = 20, fontweight='bold')
            plt.subplot(3,3,1)
            plt.imshow(matrix[:, :, slc, 0], vmin = 0, vmax = vmax_image, cmap = 'gray')
            plt.imshow(Invariant_Medians['mMD'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc], vmin = 0, vmax = 3, cmap = cDTI_cmaps['MD'], interpolation = 'nearest')
            plt.axis('off')
            plt.ylabel('Mean Diffusivity')
            plt.colorbar()
            plt.title('MD - Median of Bootstrap', fontsize = 18)

            plt.subplot(3,3,2)
            plt.imshow(matrix[:, :, slc, 0], vmin = 0, vmax = vmax_image, cmap = 'gray')
            plt.imshow(Invariant_Uncertainties['dMD'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc], vmin = 0, vmax = 3, cmap = 'plasma', interpolation = 'nearest')
            plt.axis('off')
            plt.colorbar()
            plt.title('MD - Uncertainty (dMD)', fontsize = 18)

            plt.subplot(3,3,3)
            data1_nan = Invariant_Medians['mMD'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc]
            data1_nan = data1_nan.flatten()
            data1 = data1_nan[~(np.isnan(data1_nan))]
            q25, q75 = np.percentile(data1, [25, 75])
            bin_width = 2 * (q75 - q25) * len(data1) ** (-1/3)
            bins = round((data1.max() - data1.min()) / bin_width)
            plt.hist(data1, label = 'MD',  bins = bins, density = True, histtype='bar', ec='black', alpha=0.5, color = 'red')

            data2_nan = Invariant_Uncertainties['dMD'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc]
            data2_nan = data2_nan.flatten()
            data2 = data2_nan[~(np.isnan(data2_nan))]
            q25, q75 = np.percentile(data2, [25, 75])
            bin_width = 2 * (q75 - q25) * len(data2) ** (-1/3)
            bins = round((data2.max() - data2.min()) / bin_width)    
            plt.hist(data2, label = 'dMD', bins = bins, density = True, histtype='bar', ec='black', alpha=0.5, color = 'blue')
            plt.legend()
            plt.xlim([0, 3])
            plt.title('MD LV Histogram', fontsize = 18)

            plt.subplot(3,3,4)
            plt.imshow(matrix[:, :, slc, 0], vmin = 0, vmax = vmax_image, cmap = 'gray')
            plt.imshow(Invariant_Medians['mFA'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc], vmin = 0, vmax = 1, cmap = cDTI_cmaps['FA'], interpolation = 'nearest')
            plt.axis('off')
            plt.colorbar()
            plt.title('FA - Median of Bootstrap', fontsize = 18)

            plt.subplot(3,3,5)
            plt.imshow(matrix[:, :, slc, 0], vmin = 0, vmax = vmax_image, cmap = 'gray')
            plt.imshow(Invariant_Uncertainties['dFA'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc], vmin = 0, vmax = 1, cmap = 'plasma', interpolation = 'nearest')
            plt.axis('off')
            plt.colorbar()
            plt.title('FA - Uncertainty (dFA)', fontsize = 18)

            plt.subplot(3,3,6)
            data1_nan = Invariant_Medians['mFA'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc]
            data1_nan = data1_nan.flatten()
            data1 = data1_nan[~(np.isnan(data1_nan))]
            q25, q75 = np.percentile(data1, [25, 75])
            bin_width = 2 * (q75 - q25) * len(data1) ** (-1/3)
            bins = round((data1.max() - data1.min()) / bin_width)
            plt.hist(data1, label = 'FA',  bins = bins, density = True, histtype='bar', ec='black', alpha=0.5, color = 'red')

            data2_nan = Invariant_Uncertainties['dFA'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc]
            data2_nan = data2_nan.flatten()
            data2 = data2_nan[~(np.isnan(data2_nan))]
            q25, q75 = np.percentile(data2, [25, 75])
            bin_width = 2 * (q75 - q25) * len(data2) ** (-1/3)
            bins = round((data2.max() - data2.min()) / bin_width)    
            plt.hist(data2, label = 'dFA', bins = bins, density = True, histtype='bar', ec='black', alpha=0.5, color = 'blue')
            plt.legend()
            plt.xlim([0, 1])
            plt.title('FA LV Histogram', fontsize = 18)

            plt.subplot(3,3,7)
            plt.imshow(matrix[:, :, slc, 0], vmin = 0, vmax = vmax_image, cmap = 'gray')
            plt.imshow(Invariant_Medians['mMO'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc], vmin = -1, vmax = 1, cmap = cDTI_cmaps['MO'], interpolation = 'nearest')
            plt.axis('off')
            plt.colorbar()
            plt.title('Mode - Median Bootstrap', fontsize = 18)

            plt.subplot(3,3,8)
            plt.imshow(matrix[:, :, slc, 0], vmin = 0, vmax = vmax_image, cmap = 'gray')
            plt.imshow(Invariant_Uncertainties['dMO'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc], vmin = 0, vmax = 2, cmap = 'plasma', interpolation = 'nearest')
            plt.axis('off')
            plt.colorbar()
            plt.title('Mode - Uncertainty (dMO)', fontsize = 18)

            plt.subplot(3,3,9)
            data1_nan = Invariant_Medians['mMO'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc]
            data1_nan = data1_nan.flatten()
            data1 = data1_nan[~(np.isnan(data1_nan))]
            q25, q75 = np.percentile(data1, [25, 75])
            bin_width = 2 * (q75 - q25) * len(data1) ** (-1/3)
            bins = round((data1.max() - data1.min()) / bin_width)
            plt.hist(data1, label = 'MO',  bins = bins, density = True, histtype='bar', ec='black', alpha=0.5, color = 'red')

            data2_nan = Invariant_Uncertainties['dMO'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc]
            data2_nan = data2_nan.flatten()
            data2 = data2_nan[~(np.isnan(data2_nan))]
            q25, q75 = np.percentile(data2, [25, 75])
            bin_width = 2 * (q75 - q25) * len(data2) ** (-1/3)
            bins = round((data2.max() - data2.min()) / bin_width)    
            plt.hist(data2, label = 'dMO', bins = bins, density = True, histtype='bar', ec='black', alpha=0.5, color = 'blue')
            plt.legend()
            plt.xlim([-1, 2])
            plt.title('Mode LV Histogram', fontsize = 18)

            plt.tight_layout()
            string = 'Uncertainty_cDTI_Maps_Slice_' + str(slc+1).zfill(2) + '.png'

            output_path = os.path.join(main_maps_outpath, subject_folder, Folder[flid], string)
            plt.savefig(output_path)
            plt.show()

        from scipy.io import savemat
        main_maps_outpath = '/home/ahannum/Documents/Voxel/BootstrapUltron/Uncertainty_Maps'
        if not os.path.exists(main_maps_outpath):
        # If it doesn't exist, create it
            os.makedirs(main_maps_outpath)

        data_list   = [Invariant_Medians, Invariant_Uncertainties] 
        string_list = ['Invariant_Medians', 'Invariant_Uncertainties']

        for idx in range(len(data_list)):
            string = string_list[idx]
            outpath = os.path.join(main_maps_outpath, subject_folder,  Folder[flid], string + '.mat')
            if not os.path.exists(os.path.join(main_maps_outpath,  subject_folder,  Folder[flid],)):
                # If it doesn't exist, create it
                    os.makedirs((os.path.join(main_maps_outpath,  subject_folder,  Folder[flid])))
            savemat(outpath, data_list[idx])
            
        
    overal_end_time = time.time()
    print((overal_end_time - overal_start_time)/60)
