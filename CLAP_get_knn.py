import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--survey', help='select a survey', type=int)
parser.add_argument('--itealter', help='the altered checkpoint (iteration)', type=int)
parser.add_argument('--ne', help='# experiment', type=int)

args = parser.parse_args()
survey = args.survey
itealter = args.itealter
ne = args.ne

directory_main = '/userhome/'

prix = 'pitorig_'

klist = list((1 + np.arange(40)) * 5) + list((1 + np.arange(40)) * 10 + 200) + list((1 + np.arange(20)) * 20 + 600) + list((1 + np.arange(20)) * 50 + 1000)
n_k = len(klist)



if survey == 1:
    id_train = np.load(directory_main + 'SDSS/train.npy')
    id_val = np.load(directory_main + 'SDSS/val.npy')
    id_test = np.load(directory_main + 'SDSS/test.npy')
    
    err_index = np.argwhere(id_test == 281531)
    id_test = np.delete(id_test, err_index)
    id_test_ext = np.concatenate([id_test, id_val])
    
    n_test_ext = 103305 + 20000
    n_val = 20000
    n_save = 2000
    n_train = 393219  #latentq[n_test+20000:].shape[0]
    n_all = n_test_ext + n_train

    metadata = np.load(directory_main + 'SDSS/label.npz')
    z_train = metadata['z'][id_train]
    z_test = metadata['z'][id_test_ext]
    zall = np.concatenate([z_test, z_train])
    
    if itealter == 0:
        f = np.load(directory_main + 'PZlatent/probas_net111_ADAM5_texp10_batch32_ite180000_n16s512_ebvImg_addLayer1_noAddInput_trainAll_imgnorm5_zContra_trainS2_scratch_z04_bin180_cv1ne1_.npz')
        label = 'latent1a1ei_S_'
    else:
        f = np.load(directory_main + 'PZlatent/probas_net111_ADAM5_texp10_batch32_ite180000_n16s512_ebvImg_addLayer1_noAddInput_trainAll_imgnorm5_zContra_trainS2_scratch_z04_bin180_cv1ne' + str(ne) + '__iteAlter' + str(itealter) + '_.npz')
        if ne == 1:
            label = 'latent1a1ei_S' + str(itealter) + '_'
        else:
            label = 'latent1a1ei' + str(ne) + 'ne_S' + str(itealter) + '_'
    latentq = f['latent'][:, :16]


if survey == 2:    
    only_HQ = True  
    id_train = np.load(directory_main + 'CFHT/trainc2.npy')
    id_test_ext = np.load(directory_main + 'CFHT/testc2.npy')
       
    if only_HQ:
        n_train = 100000
        id_train = id_train[:100000]
    else:
        n_train = 123434  #len(id_train)  
        
    n_test_ext = 34759
    n_val = 14759
    
    n_save = 2000
    n_all = n_test_ext + n_train        
        
    metadata = np.load(directory_main + 'CFHT/CFHTLS_catalog.npz')    
    z_train = metadata['zspec'][id_train]
    z_test = metadata['zspec'][id_test_ext]
    zall = np.concatenate([z_test, z_train])
    
    if itealter == 0:
        f = np.load(directory_main + 'PZlatent/probas_net111_ADAM5_texp10_batch32_ite120000_n16s512_ebvImg_addLayer1_noAddInput_trainAll_coeffRecon1p0_imgnorm6_zContra_trainCmixLR2_randWD_scratch_z4_bin1000_cv1ne1_.npz')
        label = 'latent1a1ei_C_12rand_' 
    else:
        f = np.load(directory_main + 'PZlatent/probas_net111_ADAM5_texp10_batch32_ite150000_n16s512_ebvImg_addLayer1_noAddInput_trainAll_coeffRecon1p0_imgnorm6_zContra_trainCmixLR2_randWD_scratch_z4_bin1000_cv1ne' + str(ne) + '__iteAlter' + str(itealter) + '_.npz')
        if ne == 1:
            label = 'latent1a1ei_C_15rand_' + str(itealter) + '_'
        else:
            label = 'latent1a1ei' + str(ne) + 'ne_C_15rand_' + str(itealter) + '_'
    latentq = f['latent'][:, :16][:n_all]

    if only_HQ:
        if 'rand' in label:
            label = label.replace('rand_', 'rand_onlyHQ_')
        else:
            label = label + 'onlyHQ_'


if survey == 3:
    survey3_onlyebv = False
    id_train = np.load(directory_main + 'KiDs_dataset/traink.npy')
    id_test_ext = np.load(directory_main + 'KiDs_dataset/testk.npy')
    
    n_train = 100000
    n_test_ext = 34147
    n_val = 14147
    n_save = 2000
    n_all = n_test_ext + n_train
    
    metadata = np.load(directory_main + 'KiDs_dataset/catalog_kids.npz')
    z_train = metadata['zspec'][id_train]
    z_test = metadata['zspec'][id_test_ext]
    zall = np.concatenate([z_test, z_train])
       
    if itealter == 0:
        f = np.load(directory_main + 'PZlatent/probas_net111_ADAM5_texp10_batch32_ite120000_n16s512_ebvmagImg_addLayer1_noAddInput_trainAll_imgnorm5_zContra_trainKIDS2rand_scratch_z3_bin800_cv1ne1_.npz')
        label = 'latent1a6ei_K_12rand_'
    else:
        if survey3_onlyebv:
            f = np.load(directory_main + 'PZlatent/probas_net111_ADAM5_texp10_batch32_ite150000_n16s512_ebvImg_addLayer1_noAddInput_trainAll_imgnorm5_zContra_trainKIDS2rand_scratch_z3_bin800_cv1ne' + str(ne) + '__iteAlter' + str(itealter) + '_.npz')
            if ne == 1:
                label = 'latent1a1ei_K_15rand_' + str(itealter) + '_'
            else:
                label = 'latent1a1ei' + str(ne) + 'ne_K_15rand_' + str(itealter) + '_'
        else:
            f = np.load(directory_main + 'PZlatent/probas_net111_ADAM5_texp10_batch32_ite150000_n16s512_ebvmagImg_addLayer1_noAddInput_trainAll_imgnorm5_zContra_trainKIDS2rand_scratch_z3_bin800_cv1ne' + str(ne) + '__iteAlter' + str(itealter) + '_.npz')
            if ne == 1:
                label = 'latent1a6ei_K_15rand_' + str(itealter) + '_'
            else:
                label = 'latent1a6ei' + str(ne) + 'ne_K_15rand_' + str(itealter) + '_'
    latentq = f['latent'][:, :16]
           
        
latentq_test = latentq[:n_test_ext]
latentq_train = latentq[n_test_ext:]
latentq_val = latentq[n_test_ext-n_val:n_test_ext]

print (n_test_ext-n_val, n_val)
print (label)



search_nn = 1

try:
    fpit = np.load(directory_main + 'PZlatent/' + label + 'texp10_knn_pit_.npz')
    search_nn = 0
    print ('do not search nn')
except:
    print ('search nn')



get_nearest_val = 1

try:
    fpit = np.load(directory_main + 'PZlatent/' + label + 'texp10_nearest_val_.npz')
    get_nearest_val = 0
    print ('do not get nearest val')
except:
    print ('get nearest val')
    


get_knn_pittrain_w_nval = 1

try:
    fpit = np.load(directory_main + 'PZlatent/' + label + 'texp10_knn_pittrain_w_nval_.npz')
    get_knn_pittrain_w_nval = 0
    print ('do not get training pit with nearest val')
except:
    print ('get training pit with nearest val')
            


compute_pitdist = 1

try:
    fpit = np.load(directory_main + 'PZlatent/' + label + prix + 'texp10_pitdist_.npz')
    compute_pitdist = 0
    print ('do not compute pitdist')
except:
    print ('compute pitdist')
        


compute_metrics = 1

try:
    fpit = np.load(directory_main + 'PZlatent/' + label + prix + 'texp10_pitdist_metrics_.npz')
    compute_metrics = 0
    print ('do not compute metrics')
except:
    print ('compute metrics')



if search_nn == 1:   # get knn, pit
    id_knn = np.zeros((n_all, n_save))
    pit_knn = np.zeros((n_all, n_k))   
        
    start = time.time()
    for i in range(n_all):
        if i % 2000 == 0: print(i, str((time.time() - start) / 60) + ' minutes')
        distsq_i = np.mean((latentq[i:i+1] - latentq_train) ** 2, 1)
        
        id_save = np.argsort(distsq_i)[:n_save]
        id_knn[i] = id_save
        
        for j in range(n_k):
            k = klist[j]
            z_i_k = z_train[id_save[:k]]
            pit_knn[i, j] = (len(z_i_k[z_i_k < zall[i]]) + 0.5 * len(z_i_k[z_i_k == zall[i]])) / k
        
    id_knn = np.cast['int32'](id_knn)
    np.savez(directory_main + 'PZlatent/' + label + 'texp10_knn_pit_', id_knn=id_knn, pit_knn=pit_knn)



if get_nearest_val == 1:
    id_nval = np.zeros(n_train)
    
    start = time.time()
    for i in range(n_train):
        if i % 2000 == 0: print(i, str((time.time() - start) / 60) + ' minutes')
        distsq_i = np.mean((latentq_train[i:i+1] - latentq_val) ** 2, 1)
        id_nval[i] = np.argmin(distsq_i)
        
    id_nval = np.cast['int32'](id_nval)
    np.savez(directory_main + 'PZlatent/' + label + 'texp10_nearest_val_', id_nval=id_nval)
    
    
    
if get_knn_pittrain_w_nval == 1:
    id_nval = np.load(directory_main + 'PZlatent/' + label + 'texp10_nearest_val_.npz')['id_nval']
    id_knn = np.load(directory_main + 'PZlatent/' + label + 'texp10_knn_pit_.npz')['id_knn']
    
    id_knn_train = id_knn[n_test_ext:]
    zval = z_test[n_test_ext-n_val:n_test_ext]
    pit_knn = np.zeros((n_train, n_k))
    
    start = time.time()
    for i in range(n_train):
        if i % 2000 == 0: print(i, str((time.time() - start) / 60) + ' minutes')
       
        id_knn_train_i = id_knn_train[i]
        zi = zval[id_nval[i]]
        for j in range(n_k):
            k = klist[j]
            z_i_k = z_train[id_knn_train_i[:k]]
            pit_knn[i, j] = (len(z_i_k[z_i_k < zi]) + 0.5 * len(z_i_k[z_i_k == zi])) / k
    
    np.savez(directory_main + 'PZlatent/' + label + 'texp10_knn_pittrain_w_nval_', pit_knn=pit_knn)
 
    

if compute_pitdist == 1:   # get pit distribution
    bins_pit_distri = 100
    pit_distri = np.zeros((n_test_ext, n_k, bins_pit_distri))
    
    fpit = np.load(directory_main + 'PZlatent/' + label + 'texp10_knn_pit_.npz')
    id_knn = fpit['id_knn']
    pit_knn = fpit['pit_knn']
    pit_knn_train = pit_knn[n_test_ext:]
        
    start = time.time()
    for i in range(n_test_ext):
        if i % 2000 == 0: print(i, str((time.time() - start) / 60) + ' minutes')
        
        pit_knn_i = pit_knn_train[id_knn[i]]   # (n_save, n_k)
        if i == 0: print (pit_knn_i.shape)
        
        for j in range(n_k):
            k = klist[j]
            pit_distri_ij = np.histogram(pit_knn_i[:k, j], bins_pit_distri, (0, 1))[0] / k
            pit_distri[i, j] = pit_distri_ij
            
    np.savez(directory_main + 'PZlatent/' + label + prix + 'texp10_pitdist_', pit_distri=pit_distri)



if compute_metrics == 1:  # get metrics
    pit_distri = np.load(directory_main + 'PZlatent/' + label + prix + 'texp10_pitdist_.npz')['pit_distri']
    bins_pit_distri = 100
    
    wasser = np.zeros((n_test_ext, n_k))
    wasser_norm = np.zeros((n_test_ext, n_k))
    tv = np.zeros((n_test_ext, n_k))
    ce = np.zeros((n_test_ext, n_k))
    mse = np.zeros((n_test_ext, n_k))
        
    flat_distri = np.ones(bins_pit_distri) / bins_pit_distri
    cdf_flat_distri = np.cumsum(flat_distri)
        
    start = time.time()
    for i in range(n_test_ext):
        if i % 2000 == 0: print(i, str((time.time() - start) / 60) + ' minutes')
        
        for j in range(n_k):
            k = klist[j]
            bins_pit_distri_ada = min(100, k)
            pit_distri_ij = pit_distri[i, j]
                        
            wasser[i, j] = np.sum(abs(np.cumsum(pit_distri_ij) - cdf_flat_distri)) / bins_pit_distri
            wasser_norm[i, j] = wasser[i, j] * np.sqrt(k) / bins_pit_distri_ada
            tv[i, j] = 0.5 * np.sum(abs(pit_distri_ij - flat_distri))
            ce[i, j] = -1 * np.sum(flat_distri * np.log(pit_distri_ij + 10**(-20)) + (1 - flat_distri) * np.log(1 - pit_distri_ij + 10**(-20)))
            mse[i, j] = np.sum((pit_distri_ij - flat_distri) ** 2)
            
    np.savez(directory_main + 'PZlatent/' + label + prix + 'texp10_pitdist_metrics_', wasser=wasser, wasser_norm=wasser_norm, tv=tv, ce=ce, mse=mse)



