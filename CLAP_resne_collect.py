import numpy as np
import time
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--survey', help='survey', type=int)
parser.add_argument('--recali_with_nval', help='recali_with_nval', type=int)
parser.add_argument('--op', help='select a collection option', type=int)

args = parser.parse_args()
survey = args.survey
recali_with_nval = args.recali_with_nval
op = args.op


directory_main = '/userhome/'
directory_io = directory_main + 'PZlatent/'

prix = 'pitorig_'


klist = list((1 + np.arange(40)) * 5) + list((1 + np.arange(40)) * 10 + 200) + list((1 + np.arange(20)) * 20 + 600) + list((1 + np.arange(20)) * 50 + 1000)
n_k = len(klist)


nelist = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
n_nelist = len(nelist)


if survey == 1:
    
    bins = 180
    zmax = 0.4
    z_prix = 'z04_bin' + str(bins)
    
    n_test_ext = 103305 + 20000
    n_test = 103305
    n_train = 393219
    
    n_all = n_test_ext + n_train
    
    id_train = np.load(directory_main + 'SDSS/train.npy')
    id_val = np.load(directory_main + 'SDSS/val.npy')
    id_test = np.load(directory_main + 'SDSS/test.npy')
    
    err_index = np.argwhere(id_test == 281531)
    id_test = np.delete(id_test, err_index)
    
    id_test_ext = np.concatenate([id_test, id_val])
    
    
    metadata = np.load(directory_main + 'SDSS/label.npz')
    z_train = metadata['z'][id_train]
    z_test = metadata['z'][id_test_ext]
    zall = np.concatenate([z_test, z_train])

    ite = 150000
    
    sprix = 'S' + str(ite)
    labelpre0 = 'latent1a1ei_S'
    f1pre0 = 'probas_net111_ADAM5_texp10_batch32_ite180000_n16s512_ebvImg_addLayer1_noAddInput_trainAll_imgnorm5_zContra_trainS2_scratch_z04_bin180_cv1ne1__iteAlter'        
    
    

if survey == 2:
    
    bins = 1000
    zmax = 4.0
    z_prix = 'z4_bin' + str(bins)
    
    n_test_ext = 34759
    n_test = 14759
    n_train = 100000
    
    n_all = n_test_ext + n_train
    
    id_train = np.load(directory_main + 'CFHT/trainc2.npy')[:100000]
    id_test_ext = np.load(directory_main + 'CFHT/testc2.npy')
    
    metadata = np.load(directory_main + 'CFHT/CFHTLS_catalog.npz')
    z_train = metadata['zspec'][id_train]
    z_test = metadata['zspec'][id_test_ext]
    zall = np.concatenate([z_test, z_train])

    ite = 120000
    
    sprix = 'C_15rand' + str(ite)
    labelpre0 = 'latent1a1ei_C_15rand_onlyHQ_'
    f1pre0 = 'probas_net111_ADAM5_texp10_batch32_ite150000_n16s512_ebvImg_addLayer1_noAddInput_trainAll_coeffRecon1p0_imgnorm6_zContra_trainCmixLR2_randWD_scratch_z4_bin1000_cv1ne1__iteAlter'




if survey == 3:
    
    bins = 800
    zmax = 3.0
    z_prix = 'z3_bin' + str(bins)
    
    n_test_ext = 34147
    n_test = 14147
    n_train = 100000
    
    n_all = n_test_ext + n_train
    
    id_train = np.load(directory_main + 'KiDs_dataset/traink.npy')
    id_test_ext = np.load(directory_main + 'KiDs_dataset/testk.npy')
    
    metadata = np.load(directory_main + 'KiDs_dataset/catalog_kids.npz')
    z_train = metadata['zspec'][id_train]
    z_test = metadata['zspec'][id_test_ext]
    zall = np.concatenate([z_test, z_train])

    ite = 120000
    
    survey3_onlyebv = False
    if survey3_onlyebv:
        sprix = 'Konlyebv_15rand' + str(ite)
        labelpre0 = 'latent1a1ei_K_15rand_'
        f1pre0 = 'probas_net111_ADAM5_texp10_batch32_ite150000_n16s512_ebvImg_addLayer1_noAddInput_trainAll_imgnorm5_zContra_trainKIDS2rand_scratch_z3_bin800_cv1ne1__iteAlter'
    else:
        sprix = 'K_15rand' + str(ite)
        labelpre0 = 'latent1a6ei_K_15rand_'
        f1pre0 = 'probas_net111_ADAM5_texp10_batch32_ite150000_n16s512_ebvmagImg_addLayer1_noAddInput_trainAll_imgnorm5_zContra_trainKIDS2rand_scratch_z3_bin800_cv1ne1__iteAlter'




yall = np.zeros((n_all, bins))
for i in range(n_all):
    z_index = max(0, min(bins - 1, int((zall[i] - 0) / zmax * bins)))
    yall[i, z_index] = 1.0
    
yall_cdf = np.cumsum(yall, 1)

    
zlist = (0.5 + np.arange(bins)) * zmax / bins



def get_wpdf_indiv(p_out):
    num, bins = p_out.shape
    wpdf_indiv = np.zeros(num)
    pcdf_axis = np.arange(bins)
    
    pcdf = np.cumsum(p_out, 1)
    
    for i in range(num):
        filt_i = (pcdf[i] > 0.16) & (pcdf[i] < 1 - 0.16)
        wpdf_indiv[i] = len(pcdf_axis[filt_i]) * zmax / bins
    return wpdf_indiv


def get_zpoints(p_out):
    num = len(p_out)
    zphot_mean = np.sum(p_out * np.expand_dims(zlist, 0), 1)

    zphot_max = np.zeros(num)
    for i in range(num):
        zphot_max[i] = zlist[np.argmax(p_out[i])]

    zphot_med = np.zeros(num)
    for i in range(num):
        zphot_med[i] = zlist[np.argmin(abs(np.cumsum(p_out[i]) - 0.5))]
    return zphot_mean, zphot_max, zphot_med


def get_moments(p_out):
    zphot_mean = np.sum(p_out * np.expand_dims(zlist, 0), 1, keepdims=True)
    variance = np.sum(p_out * (np.expand_dims(zlist, 0) - zphot_mean) ** 2, 1)
    skewness = np.sum(p_out * (np.expand_dims(zlist, 0) - zphot_mean) ** 3, 1) / variance ** 1.5
    kurtosis = np.sum(p_out * (np.expand_dims(zlist, 0) - zphot_mean) ** 4, 1) / variance ** 2
    return variance, skewness, kurtosis
    

def get_entropy_indiv(p_out):
    return -1 * np.sum(p_out * np.log(p_out + 10**(-20)), 1)



def get_avg_hom(p_set):
    n = len(p_set)
    s = np.sum(np.array([1 / p_set[i] for i in range(n)]), 0)
    ph = n / s
    ph[np.isnan(ph)] = 0
    ph = ph / np.sum(ph, 1, keepdims=True)
    return ph





if recali_with_nval == -1:
    print ('uncali')
elif recali_with_nval == 0:
    print ('recali with original train')
elif recali_with_nval == 1:
    print ('recali with nval')
elif recali_with_nval == 2:
    print ('recali with nval & train combined')

    
    
    
    
if op == 1:
    print ('raw')

    resne_collect = {}
    resne_collect['zphot_knn_mean'] = np.zeros((n_nelist, n_test_ext))  # 10
    resne_collect['pdf_knn'] = np.zeros((n_nelist, n_test_ext, bins))
    resne_collect['pit_knn'] = np.zeros((n_nelist, n_test_ext))
    resne_collect['wasser_knn'] = np.zeros((n_nelist, n_test_ext))
    
    if recali_with_nval == -1:
        resne_collect['zphot_softmax1_mean'] = np.zeros((n_nelist, n_all))
        resne_collect['zphot_softmax1_max'] = np.zeros((n_nelist, n_all))
        resne_collect['zphot_softmax1_med'] = np.zeros((n_nelist, n_all))
        resne_collect['pit_softmax1'] = np.zeros((n_nelist, n_all))
        resne_collect['wasser_softmax1'] = np.zeros((n_nelist, n_all))
    
    
    for j, ne in enumerate(nelist):
        print (j)
        
        if j == 0:
            labelpre = labelpre0
            f1pre = f1pre0
        else:
            labelpre = labelpre0.replace('ei', 'ei' + str(ne) + 'ne')
            f1pre = f1pre0.replace('cv1ne1', 'cv1ne' + str(ne))
            
            
        label = labelpre + str(ite) + '_'
        f1 = np.load(directory_main + f1pre + str(ite) + '_.npz')
    
        fpit = np.load(directory_io + label + 'texp10_knn_pit_.npz')
        id_knn = fpit['id_knn']
        pit_knn = fpit['pit_knn']
        print (id_knn.shape)
    
        fpitdist = np.load(directory_io + label + prix + 'texp10_pitdist_metrics_.npz')
        dist = fpitdist['wasser']
        print (dist.shape)
                
        if recali_with_nval == -1 or recali_with_nval == 0:
            pit_knn_train = pit_knn[n_test_ext:]
        if recali_with_nval == 1:
            pit_knn_train = np.load(directory_io + label + 'texp10_knn_pittrain_w_nval_.npz')['pit_knn']
        if recali_with_nval == 2:
            pit_knn_train = pit_knn[n_test_ext:]
            pit_knn_train2 = np.load(directory_io + label + 'texp10_knn_pittrain_w_nval_.npz')['pit_knn']
        print (pit_knn_train.shape)
            
        zdist_k = np.zeros((n_test_ext, bins))
        pit_grid = (np.arange(100) + 0.5) * 0.01
        
        start = time.time()
        for i in range(n_test_ext):
            if i % 2000 == 0: print(i, str((time.time() - start) / 60) + ' minutes')
            
            k = np.argmin(dist[i])
            kk = klist[k]
            z_i_k = z_train[id_knn[i, :kk]]
            
            zdist_ki = np.histogram(z_i_k, bins, (0, zmax))[0]
            zdist_ki = zdist_ki / np.sum(zdist_ki)
            
            if recali_with_nval >= 0:
                zdist_ki_cdf = np.cumsum(zdist_ki)
        
                pit_knn_ki = pit_knn_train[id_knn[i, :kk]][:, k]
                pit_distri_ki = np.histogram(pit_knn_ki, 100, (0, 1))[0] / kk
                
                if recali_with_nval == 2:
                    pit_knn_ki2 = pit_knn_train2[id_knn[i, :kk]][:, k]
                    pit_distri_ki2 = np.histogram(pit_knn_ki2, 100, (0, 1))[0] / kk
                    pit_distri_ki = 0.5 * (pit_distri_ki + pit_distri_ki2)
                        
                para = np.polyfit(pit_grid, pit_distri_ki, 2)
                zdist_ki = zdist_ki * np.polyval(para, zdist_ki_cdf)
            
            zdist_k[i] = zdist_ki
        zdist_k = zdist_k / np.sum(zdist_k, 1, keepdims=True) 
        
        
        resne_collect['zphot_knn_mean'][j] = np.sum(zdist_k * np.expand_dims(zlist, 0), 1)
        resne_collect['pit_knn'][j] = np.sum(zdist_k * (1 - yall_cdf[:n_test_ext] + 0.5 * yall[:n_test_ext]), 1)
        resne_collect['wasser_knn'][j] = np.sum(abs(np.cumsum(zdist_k, 1) - yall_cdf[:n_test_ext]), 1) * zmax / bins
        resne_collect['pdf_knn'][j] = zdist_k
               
        if recali_with_nval == -1:
            resne_collect['zphot_softmax1_mean'][j] = f1['zphot_mean'][0][:n_all]
            resne_collect['zphot_softmax1_max'][j] = f1['zphot_max'][0][:n_all]
            resne_collect['zphot_softmax1_med'][j] = f1['zphot_med'][0][:n_all]
            resne_collect['pit_softmax1'][j] = np.sum(f1['p_out'][:n_all] * (1 - yall_cdf + 0.5 * yall), 1)
            resne_collect['wasser_softmax1'][j] = np.sum(abs(np.cumsum(f1['p_out'][:n_all], 1) - yall_cdf), 1) * zmax / bins
    
                
            
    if recali_with_nval == -1:
        savepath = directory_io + 'resne_collect_' + sprix + '_raw_uncali'
    if recali_with_nval == 0:
        savepath = directory_io + 'resne_collect_' + sprix + '_raw'
    if recali_with_nval == 1:
        savepath = directory_io + 'resne_collect_' + sprix + '_raw_nval'
    if recali_with_nval == 2:
        savepath = directory_io + 'resne_collect_' + sprix + '_raw_trainval'
    np.savez(savepath, **resne_collect) 
    
    
    
    
if op == 2:
    alpha_gf = 0.05
    print ('fit-gf', alpha_gf)

    from scipy.ndimage import gaussian_filter as gf

    resne_collect = {}
    resne_collect['pdf_knn'] = np.zeros((n_nelist, n_test_ext, bins))
    resne_collect['zphot_mean_knn'] = np.zeros((n_nelist+2, n_test_ext))  # 10+2
    resne_collect['zphot_max_knn'] = np.zeros((n_nelist+2, n_test_ext))
    resne_collect['zphot_med_knn'] = np.zeros((n_nelist+2, n_test_ext))
    resne_collect['pit_knn'] = np.zeros((n_nelist+2, n_test_ext))
    resne_collect['wasser_knn'] = np.zeros((n_nelist+2, n_test_ext))
    resne_collect['crps_knn'] = np.zeros((n_nelist+2, n_test_ext))
    resne_collect['ce_knn'] = np.zeros((n_nelist+2, n_test_ext))
    resne_collect['width_knn'] = np.zeros((n_nelist+2, n_test_ext))
    resne_collect['entropy_knn'] = np.zeros((n_nelist+2, n_test_ext))
    resne_collect['variance_knn'] = np.zeros((n_nelist+2, n_test_ext))
    resne_collect['skewness_knn'] = np.zeros((n_nelist+2, n_test_ext))
    resne_collect['kurtosis_knn'] = np.zeros((n_nelist+2, n_test_ext))
    
    zdist_set = np.zeros((n_nelist, n_test_ext, bins))
    
        
    for j in range(n_nelist+2):
        print (j)
        
        if j < n_nelist:
            ne = nelist[j]
        
            if j == 0:
                labelpre = labelpre0
            else:
                labelpre = labelpre0.replace('ei', 'ei' + str(ne) + 'ne')                
            label = labelpre + str(ite) + '_'

            if recali_with_nval == -1:
                zdist_k = np.load(directory_io + 'probas_prefit_uncal_ADAM_wasser2ce_batch128_ite40000_' + label + 'texp10_scratch_' + z_prix + '_.npz')['p_out']    
            if recali_with_nval == 0:
                zdist_k = np.load(directory_io + 'probas_prefit_ADAM_wasser2ce_batch128_ite40000_' + label + 'texp10_scratch_' + z_prix + '_.npz')['p_out']
            if recali_with_nval == 1:
                zdist_k = np.load(directory_io + 'probas_prefit_nval_ADAM_wasser2ce_batch128_ite40000_' + label + 'texp10_scratch_' + z_prix + '_.npz')['p_out']
            if recali_with_nval == 2:
                zdist_k = np.load(directory_io + 'probas_prefit_trainval_ADAM_wasser2ce_batch128_ite40000_' + label + 'texp10_scratch_' + z_prix + '_.npz')['p_out']
            
            zdist_k = zdist_k[:n_test_ext]
            print (zdist_k.shape)
            
            variance_ini, _, _, = get_moments(zdist_k)
            sigma_j = alpha_gf * np.sqrt(variance_ini) / zmax * bins            
            for i in range(n_test_ext):
                zdist_k[i] = gf(zdist_k[i], sigma=sigma_j[i], mode='constant')
            zdist_k = zdist_k / np.sum(zdist_k, 1, keepdims=True)
            
            zdist_set[j] = zdist_k
            resne_collect['pdf_knn'][j] = zdist_k

        
        elif j == n_nelist:
            zdist_k = np.mean(zdist_set, 0)
            
                
        else:
            zdist_k = get_avg_hom(zdist_set) 
            
                
        zphot_mean, zphot_max, zphot_med = get_zpoints(zdist_k)
        variance, skewness, kurtosis = get_moments(zdist_k)
        
        resne_collect['zphot_mean_knn'][j] = zphot_mean
        resne_collect['zphot_max_knn'][j] = zphot_max
        resne_collect['zphot_med_knn'][j] = zphot_med
        resne_collect['pit_knn'][j] = np.sum(zdist_k * (1 - yall_cdf[:n_test_ext] + 0.5 * yall[:n_test_ext]), 1)
        resne_collect['wasser_knn'][j] = np.sum(abs(np.cumsum(zdist_k, 1) - yall_cdf[:n_test_ext]), 1) * zmax / bins
        resne_collect['crps_knn'][j] = np.sum((np.cumsum(zdist_k, 1) - yall_cdf[:n_test_ext]) ** 2, 1) * zmax / bins
        resne_collect['ce_knn'][j] = -1 * np.sum(yall[:n_test_ext] * np.log(zdist_k + 10**(-20)), 1)
        resne_collect['width_knn'][j] = get_wpdf_indiv(zdist_k)
        resne_collect['entropy_knn'][j] = get_entropy_indiv(zdist_k)
        resne_collect['variance_knn'][j] = variance
        resne_collect['skewness_knn'][j] = skewness
        resne_collect['kurtosis_knn'][j] = kurtosis
        
            
    if recali_with_nval == -1:
        savepath = directory_io + 'resne_collect_gf' + str(alpha_gf).replace('.', 'p') + '_' + sprix + '_prefit_uncali'
    if recali_with_nval == 0:
        savepath = directory_io + 'resne_collect_gf' + str(alpha_gf).replace('.', 'p') + '_' + sprix + '_prefit'
    if recali_with_nval == 1:
        savepath = directory_io + 'resne_collect_gf' + str(alpha_gf).replace('.', 'p') + '_' + sprix + '_prefit_nval'
    if recali_with_nval == 2:
        savepath = directory_io + 'resne_collect_gf' + str(alpha_gf).replace('.', 'p') + '_' + sprix + '_prefit_trainval'
    np.savez(savepath, **resne_collect)
   
    





