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
overal_start_time = time.time()

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


def Eigenvectors_Uncertainty_MP(Bootstrapped_DTI_Eigenvectors, slc):
    import numpy as np
    import dipy.reconst.dti as dti

    temp_E1 = Bootstrapped_DTI_Eigenvectors['E1'][:, :, slc, :, :]
    temp_E2 = Bootstrapped_DTI_Eigenvectors['E2'][:, :, slc, :, :]
    temp_E3 = Bootstrapped_DTI_Eigenvectors['E3'][:, :, slc, :, :]
    
    rows = temp_E1.shape[0]
    cols = temp_E1.shape[1]
    avgs = temp_E1.shape[3]

    dyad1 = np.zeros([rows, cols, avgs, 3, 3]) 
    dyad2 = np.zeros([rows, cols, avgs, 3, 3]) 
    dyad3 = np.zeros([rows, cols, avgs, 3, 3]) 

    theta_1 = np.zeros([rows, cols, avgs])
    theta_2 = np.zeros([rows, cols, avgs])
    theta_3 = np.zeros([rows, cols, avgs])

    CoU_1 = np.zeros([rows, cols])
    CoU_2 = np.zeros([rows, cols])
    CoU_3 = np.zeros([rows, cols])


    
    print(temp_E1.shape)
    for x in range(rows):
        for y in range(cols):
            for iters in range(avgs):
                v1 = temp_E1[x, y, :, iters]
                v2 = temp_E2[x, y, :, iters]
                v3 = temp_E3[x, y, :, iters]
                dyad1[x, y, iters, :, :] = v1[:, np.newaxis] * v1[:, np.newaxis].T
                dyad2[x, y, iters, :, :] = v2[:, np.newaxis] * v2[:, np.newaxis].T
                dyad3[x, y, iters, :, :] = v3[:, np.newaxis] * v3[:, np.newaxis].T
    dyad1_mean = np.mean(dyad1, axis = 2)
    dyad2_mean = np.mean(dyad2, axis = 2)
    dyad3_mean = np.mean(dyad3, axis = 2)

    valtemp1, vectemp1 = dti.decompose_tensor(dyad1_mean)
    Psi1 = vectemp1[:, :, :, 0]
    valtemp2, vectemp2 = dti.decompose_tensor(dyad2_mean)
    Psi2 = vectemp2[:, :, :, 0]
    valtemp3, vectemp3 = dti.decompose_tensor(dyad3_mean)
    Psi3 = vectemp3[:, :, :, 0]

    beta_1_1 = valtemp1[:, :, 0]
    beta_2_1 = valtemp1[:, :, 1]
    beta_3_1 = valtemp1[:, :, 2]
    kappa_1  = (1 - np.sqrt((beta_2_1 + beta_3_1) / (2 * beta_1_1)))

    beta_1_2 = valtemp2[:, :, 0]
    beta_2_2 = valtemp2[:, :, 1]
    beta_3_2 = valtemp2[:, :, 2]
    kappa_2  = (1 - np.sqrt((beta_2_2 + beta_3_2) / (2 * beta_1_2)))

    beta_1_3 = valtemp3[:, :, 0]
    beta_2_3 = valtemp3[:, :, 1]
    beta_3_3 = valtemp3[:, :, 2]
    kappa_3  = (1 - np.sqrt((beta_2_3 + beta_3_3) / (2 * beta_1_3)))

    Dyadic_Tensor = dict()
    Dyadic_Tensor['E1'] = dyad1
    Dyadic_Tensor['E2'] = dyad2
    Dyadic_Tensor['E3'] = dyad3

    Dyadic_Mean   = dict()
    Dyadic_Mean['Psi1'] = Psi1
    Dyadic_Mean['Psi2'] = Psi2
    Dyadic_Mean['Psi3'] = Psi3

    Coherence     = dict()
    Coherence['K1'] = kappa_1
    Coherence['K2'] = kappa_2
    Coherence['K3'] = kappa_3

    for x in range(rows):
        for y in range(cols):
            for iters in range(avgs):
                v1        = temp_E1[x, y, :, iters]
                angle_1   = np.arccos(np.dot(Psi1[x, y, :], v1)) * 180 / np.pi
                if angle_1 <= 90:
                    tmp_mask1 = 0
                else:
                    tmp_mask1 = 180
                v2        = temp_E2[x, y, :, iters]
                angle_2   = np.arccos(np.dot(Psi2[x, y, :], v2)) * 180 / np.pi
                if angle_2 <= 90:
                    tmp_mask2 = 0
                else:
                    tmp_mask2 = 180

                v3        = temp_E3[x, y, :, iters]
                angle_3   = np.arccos(np.dot(Psi3[x, y, :], v3)) * 180 / np.pi
                if angle_3 <= 90:
                    tmp_mask3 = 0
                else:
                    tmp_mask3 = 180

                theta_1[x, y, iters] = np.abs(angle_1 - tmp_mask1)
                theta_2[x, y, iters] = np.abs(angle_2 - tmp_mask2)
                theta_3[x, y, iters] = np.abs(angle_3 - tmp_mask3)


            theta_1_sort = theta_1[x, y, :]
            theta_1_sort = np.sort(theta_1_sort.flatten() )
            pos95_1      = int(np.round(0.95 * len(theta_1_sort))) - 1
            CoU_1[x, y]  = theta_1_sort[int(pos95_1)]

            theta_2_sort = theta_2[x, y, :]
            theta_2_sort = np.sort(theta_2_sort.flatten() )
            pos95_2      = int(np.round(0.95 * len(theta_2_sort))) - 1
            CoU_2[x, y]  = theta_2_sort[int(pos95_2)]

            theta_3_sort = theta_3[x, y, :]
            theta_3_sort = np.sort(theta_3_sort.flatten() )
            pos95_3      = int(np.round(0.95 * len(theta_3_sort))) - 1
            CoU_3[x, y]  = theta_3_sort[int(pos95_3)]
    Difference_Angle   = dict()
    Difference_Angle['theta1'] = theta_1
    Difference_Angle['theta2'] = theta_2
    Difference_Angle['theta3'] = theta_3

    Cone_of_Uncertainty = dict()
    Cone_of_Uncertainty['dE1'] = CoU_1
    Cone_of_Uncertainty['dE2'] = CoU_2
    Cone_of_Uncertainty['dE3'] = CoU_3

    return [Dyadic_Tensor, Dyadic_Mean, Coherence, Difference_Angle, Cone_of_Uncertainty]



main_data_path         = '/home/ahannum/Documents/Voxel/BootstrapUltron/'
main_image_path        = '/home/ahannum/Documents/Voxel/BootstrapUltron/'
main_segmentation_path = '/home/ahannum/Documents/Voxel/BootstrapUltron/'
volunteers        = ['V002','V003','V004','V005','V006','V007','V008','V009','V010','V011',]

for subject_folder in volunteers:

    #DTI_folder  = 'cDTI'
    Folder       = ['vol_2.0res_3.0sl','vol_2.0res_5.0sl','vol_2.0res_8.0sl', 'vol_2.5res_3.0sl','vol_2.5res_5.0sl','vol_2.5res_8.0sl', 'vol_3.0res_3.5sl','vol_3.0res_5.5sl','vol_3.0res_8.0sl']
    # Folder      = 'DiVO_06_10'
    Data_Folder = '07_Bootstrapping'
    Evecs_Folder = '07_Bootstrapping'


    for flid in range(len(Folder)):
        print(glob.glob(os.path.join(main_segmentation_path, subject_folder, Folder[flid], '06_cDTI_Maps/Interpolated_mask.mat')))
        inpath_1 = glob.glob(os.path.join(main_segmentation_path, subject_folder, Folder[flid], '06_cDTI_Maps/Interpolated_mask.mat'))[0]
        print('Path to NRRD images:  ' + inpath_1)
        NRRD = loadmat(os.path.join(inpath_1))['Mask']
        

        inpath_2 = glob.glob(os.path.join(main_image_path, subject_folder, Folder[flid], Data_Folder, 'Registered_1000_Bootstraps.header'))[0]
        print('Path to header:       ' + inpath_2)
        Header = Header_Reader(inpath_2)

        inpath_3 = glob.glob(os.path.join(main_image_path, subject_folder, Folder[flid], Data_Folder, 'Registered_1000_Bootstraps.nii'))[0]
        print('Path to NifTi images: ' + inpath_3)
        NifTi, affine_matrix, NifTi_VoxRes = load_nifti(inpath_3, return_voxsize = True)

        Header['X Resolution'] = NifTi_VoxRes[0]
        Header['Y Resolution'] = NifTi_VoxRes[1]
        Header['Z Resolution'] = NifTi_VoxRes[2]
        print(Header)

        inpath_4 = glob.glob(os.path.join(main_data_path, subject_folder,  Folder[flid], Evecs_Folder, 'DTI_Eigenvectors_1000_Bootstraps.mat'))[0]
        print('Path to Eigenvectors: ' + inpath_4)
        mat = scipy.io.loadmat(inpath_4)
        #mat = scipy.io.loadmat(inpath_4)

        # matricies.append(NifTi)
        # bvalues.append(bvals)
        # bvectors.append(bvecs)
        # labels.append('Blip Down')

        # del inpath_3, inpath_4, inpath_5, NifTi, affine_matrix, NifTi_VoxRes, bvals, bvecs

        Bootstrapped_E1 = mat['E1'][:, :, :, :, :]
        Bootstrapped_E2 = mat['E2'][:, :, :, :, :]
        Bootstrapped_E3 = mat['E3'][:, :, :, :, :]

        Bootstrapped_DTI_Eigenvectors = dict()
        Bootstrapped_DTI_Eigenvectors['E1'] = Bootstrapped_E1
        Bootstrapped_DTI_Eigenvectors['E2'] = Bootstrapped_E2
        Bootstrapped_DTI_Eigenvectors['E3'] = Bootstrapped_E3

        from joblib import Parallel, delayed
        import multiprocessing

        # num_cores = 3
        num_cores = 3
        start = time.time()

        rows = NRRD.shape[0]
        cols = NRRD.shape[1]
        slcs = NRRD.shape[2]

        inputs = range(NRRD.shape[2])

        results = Parallel(n_jobs=num_cores)(delayed(Eigenvectors_Uncertainty_MP)(Bootstrapped_DTI_Eigenvectors, slcs) for slcs in inputs)
        end = time.time()
        print(end - start) 

        rows = results[0][0]['E1'].shape[0]
        cols = results[0][0]['E1'].shape[1]
        slcs = len(results)
        avgs = results[0][0]['E1'].shape[2]

        dyad1_matrix  = np.zeros([rows, cols, slcs, avgs, 3, 3])
        dyad2_matrix  = np.zeros([rows, cols, slcs, avgs, 3, 3])
        dyad3_matrix  = np.zeros([rows, cols, slcs, avgs, 3, 3])
        Psi1_matrix   = np.zeros([rows, cols, slcs, 3])
        Psi2_matrix   = np.zeros([rows, cols, slcs, 3])
        Psi3_matrix   = np.zeros([rows, cols, slcs, 3])
        kappa1_matrix = np.zeros([rows, cols, slcs])
        kappa2_matrix = np.zeros([rows, cols, slcs])
        kappa3_matrix = np.zeros([rows, cols, slcs])
        theta1_matrix = np.zeros([rows, cols, slcs, avgs])
        theta2_matrix = np.zeros([rows, cols, slcs, avgs])
        theta3_matrix = np.zeros([rows, cols, slcs, avgs])
        CoU1_matrix   = np.zeros([rows, cols, slcs])
        CoU2_matrix   = np.zeros([rows, cols, slcs])
        CoU3_matrix   = np.zeros([rows, cols, slcs])

        for slc in range(slcs):
            dyad1_matrix[:, :, slc, :, :, :] = results[slc][0]['E1']
            dyad2_matrix[:, :, slc, :, :, :] = results[slc][0]['E2']
            dyad3_matrix[:, :, slc, :, :, :] = results[slc][0]['E3']
            Psi1_matrix[:, :, slc, :]        = results[slc][1]['Psi1']
            Psi2_matrix[:, :, slc, :]        = results[slc][1]['Psi2']
            Psi3_matrix[:, :, slc, :]        = results[slc][1]['Psi3']
            kappa1_matrix[:, :, slc]         = results[slc][2]['K1']
            kappa2_matrix[:, :, slc]         = results[slc][2]['K2']
            kappa3_matrix[:, :, slc]         = results[slc][2]['K3']
            theta1_matrix[:, :, slc, :]      = results[slc][3]['theta1']
            theta2_matrix[:, :, slc, :]      = results[slc][3]['theta2']
            theta3_matrix[:, :, slc, :]      = results[slc][3]['theta3']
            CoU1_matrix[:, :, slc]           = results[slc][4]['dE1']
            CoU2_matrix[:, :, slc]           = results[slc][4]['dE2']
            CoU3_matrix[:, :, slc]           = results[slc][4]['dE3']

        Difference_Angle   = dict()
        Difference_Angle['theta1'] = theta1_matrix
        Difference_Angle['theta2'] = theta2_matrix
        Difference_Angle['theta3'] = theta3_matrix

        Cone_of_Uncertainty = dict()
        Cone_of_Uncertainty['dE1'] = CoU1_matrix
        Cone_of_Uncertainty['dE2'] = CoU2_matrix
        Cone_of_Uncertainty['dE3'] = CoU3_matrix

        Dyadic_Tensor = dict()
        Dyadic_Tensor['E1'] = dyad1_matrix
        Dyadic_Tensor['E2'] = dyad2_matrix
        Dyadic_Tensor['E3'] = dyad3_matrix

        Dyadic_Mean   = dict()
        Dyadic_Mean['Psi1'] = Psi1_matrix
        Dyadic_Mean['Psi2'] = Psi2_matrix
        Dyadic_Mean['Psi3'] = Psi3_matrix

        Coherence     = dict()
        Coherence['K1'] = kappa1_matrix
        Coherence['K2'] = kappa2_matrix
        Coherence['K3'] = kappa3_matrix

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
            fig = plt.figure(figsize = (30, 15), dpi= 100)
            fig.patch.set_facecolor('white')
            fig.suptitle('Uncertainty Bootstrap cDTI Maps for Slice %i' %(slc + 1), fontsize = 20, fontweight='bold')
            plt.subplot(3,5,1)
            plt.imshow(matrix[:, :, slc, 0], vmin = 0, vmax = vmax_image, cmap = 'gray')
            plt.colorbar()
            tmp1 = np.expand_dims(myocardial_mask_smoothed_nan[:, :, slc], axis = 2)
            tmp2 = abs(Dyadic_Mean['Psi1'][:, :, slc, :])
            tmp3 = np.concatenate((tmp2, tmp1), axis = 2)
            plt.imshow(tmp3)
            plt.axis('off')
            plt.title('$E_1$ Dyadic Mean  ($\u03A8_1$)', fontsize = 18)

            plt.subplot(3,5,2)
            plt.imshow(matrix[:, :, slc, 0], vmin = 0, vmax = vmax_image, cmap = 'gray')
            plt.imshow(Coherence['K1'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc], vmin = 0, vmax = 1, cmap = 'viridis', interpolation = 'nearest')
            plt.axis('off')
            plt.colorbar()
            plt.title('Coherence ($\u03BA_1$)', fontsize = 18)

            plt.subplot(3,5,3)
            data1_nan = Coherence['K1'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc]
            data1_nan = data1_nan.flatten()
            data1 = data1_nan[~(np.isnan(data1_nan))]
            q25, q75 = np.percentile(data1, [25, 75])
            bin_width = 2 * (q75 - q25) * len(data1) ** (-1/3)
            bins = round((data1.max() - data1.min()) / bin_width)
            plt.hist(data1, label = '$\u03BA_1$',  bins = bins, density = True, histtype='bar', ec='black', alpha=0.5, color = 'red')
            plt.title('$\u03BA_1$ LV Histogram', fontsize = 18)
            plt.xlim([0, 1])

            plt.subplot(3,5,4)
            plt.imshow(matrix[:, :, slc, 0], vmin = 0, vmax = vmax_image, cmap = 'gray')
            plt.imshow(Cone_of_Uncertainty['dE1'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc], vmin = 0, vmax = 90, cmap = 'plasma', interpolation = 'nearest')
            plt.axis('off')
            plt.colorbar()
            plt.title('$E_1$ Cone of Uncertainty (d$E_1$)', fontsize = 18)

            plt.subplot(3,5,5)
            data1_nan = Cone_of_Uncertainty['dE1'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc]
            data1_nan = data1_nan.flatten()
            data1 = data1_nan[~(np.isnan(data1_nan))]
            q25, q75 = np.percentile(data1, [25, 75])
            bin_width = 2 * (q75 - q25) * len(data1) ** (-1/3)
            bins = round((data1.max() - data1.min()) / bin_width)
            plt.hist(data1, label = 'd$E_1$',  bins = bins, density = True, histtype='bar', ec='black', alpha=0.5, color = 'red')
            plt.title('d$E_1$ LV Histogram', fontsize = 18)
            plt.xlim([0, 90])

            plt.subplot(3,5,6)
            plt.imshow(matrix[:, :, slc, 0], vmin = 0, vmax = vmax_image, cmap = 'gray')
            plt.colorbar()
            tmp1 = np.expand_dims(myocardial_mask_smoothed_nan[:, :, slc], axis = 2)
            tmp2 = abs(Dyadic_Mean['Psi2'][:, :, slc, :])
            tmp3 = np.concatenate((tmp2, tmp1), axis = 2)
            plt.imshow(tmp3)
            plt.axis('off')
            plt.title('$E_2$ Dyadic Mean  ($\u03A8_2$)', fontsize = 18)

            plt.subplot(3,5,7)
            plt.imshow(matrix[:, :, slc, 0], vmin = 0, vmax = vmax_image, cmap = 'gray')
            plt.imshow(Coherence['K2'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc], vmin = 0, vmax = 1, cmap = 'viridis', interpolation = 'nearest')
            plt.axis('off')
            plt.colorbar()
            plt.title('Coherence ($\u03BA_2$)', fontsize = 18)

            plt.subplot(3,5,8)
            data1_nan = Coherence['K2'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc]
            data1_nan = data1_nan.flatten()
            data1 = data1_nan[~(np.isnan(data1_nan))]
            q25, q75 = np.percentile(data1, [25, 75])
            bin_width = 2 * (q75 - q25) * len(data1) ** (-1/3)
            bins = round((data1.max() - data1.min()) / bin_width)
            plt.hist(data1, label = '$\u03BA_2$',  bins = bins, density = True, histtype='bar', ec='black', alpha=0.5, color = 'red')
            plt.title('$\u03BA_2$ LV Histogram', fontsize = 18)
            plt.xlim([0, 1])

            plt.subplot(3,5,9)
            plt.imshow(matrix[:, :, slc, 0], vmin = 0, vmax = vmax_image, cmap = 'gray')
            plt.imshow(Cone_of_Uncertainty['dE2'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc], vmin = 0, vmax = 90, cmap = 'plasma', interpolation = 'nearest')
            plt.axis('off')
            plt.colorbar()
            plt.title('$E_2$ Cone of Uncertainty (d$E_2$)', fontsize = 18)

            plt.subplot(3,5,10)
            data1_nan = Cone_of_Uncertainty['dE2'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc]
            data1_nan = data1_nan.flatten()
            data1 = data1_nan[~(np.isnan(data1_nan))]
            q25, q75 = np.percentile(data1, [25, 75])
            bin_width = 2 * (q75 - q25) * len(data1) ** (-1/3)
            bins = round((data1.max() - data1.min()) / bin_width)
            plt.hist(data1, label = 'd$E_2$',  bins = bins, density = True, histtype='bar', ec='black', alpha=0.5, color = 'red')
            plt.title('d$E_2$ LV Histogram', fontsize = 18)
            plt.xlim([0, 90])

            plt.subplot(3,5,11)
            plt.imshow(matrix[:, :, slc, 0], vmin = 0, vmax = vmax_image, cmap = 'gray')
            plt.colorbar()
            tmp1 = np.expand_dims(myocardial_mask_smoothed_nan[:, :, slc], axis = 2)
            tmp2 = abs(Dyadic_Mean['Psi3'][:, :, slc, :])
            tmp3 = np.concatenate((tmp2, tmp1), axis = 2)
            plt.imshow(tmp3)
            plt.axis('off')
            plt.title('$E_3$ Dyadic Mean  ($\u03A8_3$)', fontsize = 18)

            plt.subplot(3,5,12)
            plt.imshow(matrix[:, :, slc, 0], vmin = 0, vmax = vmax_image, cmap = 'gray')
            plt.imshow(Coherence['K3'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc], vmin = 0, vmax = 1, cmap = 'viridis', interpolation = 'nearest')
            plt.axis('off')
            plt.colorbar()
            plt.title('Coherence ($\u03BA_3$)', fontsize = 18)

            plt.subplot(3,5,13)
            data1_nan = Coherence['K3'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc]
            data1_nan = data1_nan.flatten()
            data1 = data1_nan[~(np.isnan(data1_nan))]
            q25, q75 = np.percentile(data1, [25, 75])
            bin_width = 2 * (q75 - q25) * len(data1) ** (-1/3)
            bins = round((data1.max() - data1.min()) / bin_width)
            plt.hist(data1, label = '$\u03BA_3$',  bins = bins, density = True, histtype='bar', ec='black', alpha=0.5, color = 'red')
            plt.title('$\u03BA_3$ LV Histogram', fontsize = 18)
            plt.xlim([0, 1])

            plt.subplot(3,5,14)
            plt.imshow(matrix[:, :, slc, 0], vmin = 0, vmax = vmax_image, cmap = 'gray')
            plt.imshow(Cone_of_Uncertainty['dE3'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc], vmin = 0, vmax = 90, cmap = 'plasma', interpolation = 'nearest')
            plt.axis('off')
            plt.colorbar()
            plt.title('$E_3$ Cone of Uncertainty (d$E_3$)', fontsize = 18)

            plt.subplot(3,5,15)
            data1_nan = Cone_of_Uncertainty['dE3'][:, :, slc] * myocardial_mask_smoothed_nan[:, :, slc]
            data1_nan = data1_nan.flatten()
            data1 = data1_nan[~(np.isnan(data1_nan))]
            q25, q75 = np.percentile(data1, [25, 75])
            bin_width = 2 * (q75 - q25) * len(data1) ** (-1/3)
            bins = round((data1.max() - data1.min()) / bin_width)
            plt.hist(data1, label = 'd$E_3$',  bins = bins, density = True, histtype='bar', ec='black', alpha=0.5, color = 'red')
            plt.title('d$E_3$ LV Histogram', fontsize = 18)
            plt.xlim([0, 90])


            plt.tight_layout()
            string = 'Vector_Uncertainty_cDTI_Maps_Slice_' + str(slc+1).zfill(2) + '.png'

            output_path = os.path.join(main_maps_outpath, subject_folder, Folder[flid], string)
            
            plt.savefig(output_path)
            plt.show()

        from scipy.io import savemat
        main_maps_outpath = '/home/ahannum/Documents/Voxel/BootstrapUltron/Uncertainty_Maps'
        if not os.path.exists(main_maps_outpath):
        # If it doesn't exist, create it
            os.makedirs(main_maps_outpath)

        data_list   = [Dyadic_Mean, Coherence, Cone_of_Uncertainty] 
        string_list = ['Vector_Dyadic_Means', 'Vector_Coherences', 'Vector_Uncertainties']

        for idx in range(len(data_list)):
            string = string_list[idx]
            outpath = os.path.join(main_maps_outpath, subject_folder,  Folder[flid], string + '.mat')
            if not os.path.exists(os.path.join(main_maps_outpath,  subject_folder,  Folder[flid],)):
                # If it doesn't exist, create it
                    os.makedirs((os.path.join(main_maps_outpath,  subject_folder,  Folder[flid])))
            savemat(outpath, data_list[idx])



        del inpath_1, inpath_2, inpath_3, NifTi, affine_matrix, NifTi_VoxRes, inpath_4, mat
        del Bootstrapped_E1, Bootstrapped_E2, Bootstrapped_E3, Bootstrapped_DTI_Eigenvectors
        del num_cores, start, rows, cols, slcs, avgs, inputs, results, end
        del dyad1_matrix, dyad2_matrix, dyad3_matrix, Psi1_matrix, Psi2_matrix, Psi3_matrix
        del kappa1_matrix, kappa2_matrix, kappa3_matrix, theta1_matrix, theta2_matrix, theta3_matrix
        del CoU1_matrix, CoU2_matrix, CoU3_matrix
        del Difference_Angle, Cone_of_Uncertainty, Dyadic_Tensor, Dyadic_Mean, Coherence
        del matrix, myocardial_mask_smoothed, myocardial_mask_smoothed_nan
        del main_maps_outpath, data_list, string_list
        #del K1_List, K2_List, K3_List, dE1_List, dE2_List, dE3_List, Slice_List
        #del df_K1, df_K2, df_K3, df_dE1, df_dE2, df_dE3, df_list

    overal_end_time = time.time()
    print((overal_end_time - overal_start_time)/60)

