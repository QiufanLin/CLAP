import numpy as np


def img_norm(img):
    index_neg = img < 0
    index_pos = img > 0
    img[index_pos] = np.log(img[index_pos] + 1.0)
    img[index_neg] = -np.log(-img[index_neg] + 1.0)
    return img


def img_reshape(a):
    mode = np.random.random()
    if mode < 0.25: a = np.rot90(a, 1)
    elif mode < 0.50: a = np.rot90(a, 2)
    elif mode < 0.75: a = np.rot90(a, 3)
    else: pass
        
    mode = np.random.random()
    if mode < 0.5: a = np.flip(a, 0)
    else: pass            
    return a


def img_morph_aug(a):
    mode = np.random.random()
    if mode < 0.25: a = np.rot90(a, 1)
    elif mode < 0.50: a = np.rot90(a, 2)
    elif mode < 0.75: a = np.rot90(a, 3)
        
    if mode > 0.75: mode = 0
    else: mode = np.random.random()
    if mode < 0.5: a = np.flip(a, 0)
    return a



class LoadSDSS:
    def __init__(self, directory_main, index1_ini, add_inputs):
        self.directory_main = directory_main
        self.directory_input = directory_main + 'PZlatent/'
        self.index1_ini = index1_ini
        self.add_inputs = add_inputs

    def load_data(self):
        labels_ini = ['z', 'EBV', 'zphot', 'dered_petro_u', 'dered_petro_g', 'dered_petro_r', 'dered_petro_i', 'dered_petro_z',
                      'petroMagErr_u', 'petroMagErr_g', 'petroMagErr_r', 'petroMagErr_i', 'petroMagErr_z']        
        labels = ['z', 'ebv', 'zphot', 'umag', 'gmag', 'rmag', 'imag', 'zmag',
                      'err_umag', 'err_gmag', 'err_rmag', 'err_imag', 'err_zmag',
                      'ug', 'gr', 'ri', 'iz', 'err_ug', 'err_gr', 'err_ri', 'err_iz']        
        metadata = np.load(self.directory_main + 'SDSS/label.npz')
                
        properties = {}
        for i in range(len(labels_ini)):
            properties[labels[i]] = metadata[labels_ini[i]]
            
        properties['umag'] = properties['umag'] + metadata['extinction_u']
        properties['gmag'] = properties['gmag'] + metadata['extinction_g']
        properties['rmag'] = properties['rmag'] + metadata['extinction_r']
        properties['imag'] = properties['imag'] + metadata['extinction_i']
        properties['zmag'] = properties['zmag'] + metadata['extinction_z']  
        
        for i in range(4):
            properties[labels[i-8]] = properties[labels[i-18]] - properties[labels[i-17]]   # color
            properties[labels[i-4]] = np.sqrt((properties[labels[i-13]]) ** 2 + (properties[labels[i-12]]) ** 2)   # err_color
        
        for i in range(5): # clean mag
            filt = np.isnan(properties[labels[i-18]]) | (properties[labels[i-18]] < 0)
            properties[labels[i-18]][filt] = np.max(properties[labels[i-18]])
            properties[labels[i-13]][filt] = 10**5
        for i in range(4): # clean color
            filt = np.isnan(properties[labels[i-8]]) | (properties[labels[i-8]] < 0)
            properties[labels[i-8]][filt] = np.max(properties[labels[i-8]])
            properties[labels[i-4]][filt] = 10**5

        obj_train = np.load(self.directory_main + 'SDSS/train.npy')
        obj_val = np.load(self.directory_main + 'SDSS/val.npy')
        obj_test = np.load(self.directory_main + 'SDSS/test.npy')
        
        obj_all = np.arange(len(properties['z']))
        mask = np.zeros(len(properties['z']))
        mask[obj_train] = 1.0
        mask[obj_val] = 1.0
        mask[obj_test] = 1.0
        obj_other = obj_all[mask == 0]
        
        err_index = np.argwhere(obj_test == 281531)
        obj_test = np.delete(obj_test, err_index)
        obj_test = np.concatenate([obj_test, obj_val])

        
        if self.add_inputs > 1:
            properties['inputadd'] = np.stack([properties['ebv'], properties['umag'], properties['gmag'], properties['rmag'], properties['imag'], properties['zmag']], 1)    
        else:
            properties['inputadd'] = np.expand_dims(properties['ebv'], -1)
            
            
        print ('Load SDSS metadata')
        print ('SDSS Training,Test,Val:', len(obj_train), len(obj_test), len(obj_val))
        print ('#####')
        return properties, obj_train, obj_test, obj_other
 



class LoadCFHTLS:
    def __init__(self, directory_main, index1_ini, add_inputs, z_max):
        self.directory_main = directory_main
        self.directory_input = directory_main + 'PZlatent/'
        self.index1_ini = index1_ini
        self.add_inputs = add_inputs
        self.z_max = z_max


    def load_data(self):
        labels_ini = ['ZSPEC', 'EBV', 'zphot', 'U', 'G', 'R', 'I', 'Z',
                      'Uerr', 'Gerr', 'Rerr', 'Ierr', 'Zerr']        
        labels = ['z', 'ebv', 'zphot', 'umag', 'gmag', 'rmag', 'imag', 'zmag',
                      'err_umag', 'err_gmag', 'err_rmag', 'err_imag', 'err_zmag',
                      'ug', 'gr', 'ri', 'iz', 'err_ug', 'err_gr', 'err_ri', 'err_iz']        
        properties = {}
        
        for j in range(20):
            fi = np.load(self.directory_main + 'CFHT/CUBE_WD_ZHQ_' + ('000' + str(j+1))[-4:] + '.npz')
            metadata = fi['info']
            cube = fi['cube']
            if j == 0:
                properties['imgs'] = []
                properties['field'] = []
            properties['imgs'].append(cube)
            properties['field'].append(metadata['FIELD'])
                
            for i in range(len(labels_ini)):
                if j == 0: 
                    properties[labels[i]] = []
                properties[labels[i]].append(metadata[labels_ini[i]])
        
        z_ini = np.concatenate(properties['z'])
        field_ini = np.concatenate(properties['field'])
        obj_all = np.arange(len(z_ini))[z_ini < self.z_max]  # 134759

        obj_all_shuffle = obj_all[self.index1_ini[self.index1_ini < len(obj_all)]]
        obj_test = obj_all_shuffle[:34759]
        obj_train = obj_all_shuffle[34759:]
        
        obj_trainW = np.array([obj for obj in obj_train if 'W' in str(field_ini[obj])])
        obj_trainD = np.array([obj for obj in obj_train if 'D' in str(field_ini[obj])])
        

        labels_ini = ['ZSPEC', 'EBV', 'ZPHOT', 'U', 'G', 'R', 'Ip_2', 'Z',
                      'ERRU', 'ERRG', 'ERRR', 'ERRIp_2', 'ERRZ']
        
        for j in range(2):
            fi = np.load(self.directory_main + 'CFHT/CUBE_WD_ZLR_' + ('000' + str(j+1))[-4:] + '.npz')                
            metadata = fi['info']
            cube = fi['cube']
            properties['imgs'].append(cube)
            properties['field'].append(metadata['FIELD'])

            for i in range(len(labels_ini)):
                properties[labels[i]].append(metadata[labels_ini[i]])

        z_ini_add = np.concatenate([properties['z'][-2], properties['z'][-1]])
        obj_add = np.arange(len(z_ini_add))[z_ini_add < self.z_max] + len(z_ini)
        obj_train = np.array(list(obj_train) + list(obj_add))

        
        property_keys = list(properties.keys())
        for j in range(len(property_keys)):
            properties[property_keys[j]] = np.concatenate(properties[property_keys[j]], 0)
    
        for i in range(4):
            properties[labels[i-8]] = properties[labels[i-18]] - properties[labels[i-17]]   # color
            properties[labels[i-4]] = np.sqrt((properties[labels[i-13]]) ** 2 + (properties[labels[i-12]]) ** 2)   # err_color
            
        for i in range(5): # clean mag
            filt = np.isnan(properties[labels[i-18]]) | (properties[labels[i-18]] < 0)
      #      properties[labels[i-18]][filt] = np.max(properties[labels[i-18]][properties[labels[i-18]] > -9000])
            properties[labels[i-18]][filt] = 0.0
            properties[labels[i-13]][filt] = 10**5
        for i in range(4): # clean color
            filt = np.isnan(properties[labels[i-8]]) | (properties[labels[i-8]] < 0)
            properties[labels[i-8]][filt] = np.max(properties[labels[i-8]][properties[labels[i-8]] > -9000])
            properties[labels[i-4]][filt] = 10**5

        if self.add_inputs > 1:
            properties['inputadd'] = np.stack([properties['ebv'], properties['umag'], properties['gmag'], properties['rmag'], properties['imag'], properties['zmag']], 1)    
        else:
            properties['inputadd'] = np.expand_dims(properties['ebv'], -1)


        obj_all = np.arange(len(properties['z']))
        mask = np.zeros(len(properties['z']))
        mask[obj_train] = 1.0
        mask[obj_test] = 1.0
        obj_other = obj_all[mask == 0]
        

        print ('Load CFHT metadata')
   #     print ('W,D,LR:', len(obj_W), len(obj_D))  #, len(obj_add))     
        print ('Training,Test:', len(obj_train), len(obj_test))
        print ('#####')
        return properties, obj_train, obj_test, obj_other, obj_trainD, obj_trainW



class LoadKIDS:
    def __init__(self, directory_main, index1_ini, add_inputs):
        self.directory_main = directory_main
        self.directory_input = directory_main + 'PZlatent/'
        self.index1_ini = index1_ini
        self.add_inputs = add_inputs


    def load_data(self):
        labels_ini = ['zspec', 'EBV', 'u', 'g', 'r', 'i', 'Z', 'Y', 'J', 'H', 'Ks']        
        labels = ['z', 'ebv', 'umag', 'gmag', 'rmag', 'imag', 'Zmag', 'Ymag', 'Jmag', 'Hmag', 'Ksmag']
        
        metadata = np.load(self.directory_main + 'KiDs_dataset/catalog_kids.npz')
        
        
        properties = {}
        for i in range(len(labels_ini)):
            properties[labels[i]] = metadata[labels_ini[i]]
            

        obj_all = np.arange(len(metadata['zspec']))  # 134147

        obj_all_shuffle = obj_all[self.index1_ini[self.index1_ini < len(obj_all)]]
        obj_test = obj_all_shuffle[:34147]
        obj_train = obj_all_shuffle[34147:]
        obj_other = []
        
   #     np.save(self.directory_main + 'KiDs_dataset/traink', obj_train)
   #     np.save(self.directory_main + 'KiDs_dataset/testk', obj_test)

        if self.add_inputs == 3:
            properties['inputadd'] = np.stack([properties['ebv'], properties['umag'], properties['gmag'], properties['rmag'], properties['imag'], properties['Zmag'], properties['Ymag'], properties['Jmag'], properties['Hmag'], properties['Ksmag']], 1)
            print ('add all mag')
        elif self.add_inputs == 2:
            properties['inputadd'] = np.stack([properties['ebv'], properties['Zmag'], properties['Ymag'], properties['Jmag'], properties['Hmag'], properties['Ksmag']], 1)
            print ('add mag')
        else:
            properties['inputadd'] = np.expand_dims(properties['ebv'], -1)
            
        print ('Load KIDS metadata')
        print ('KIDS Training,Test,Val:', len(obj_train), len(obj_test))
        print ('#####')
        return properties, obj_train, obj_test, obj_other
    
    
    
    

class GetData:
    def __init__(self, directory_main, directory_input, index1_ini, texp, size_latent_main, size_latent_ext, img_size, channels, bins, z_min, z_max, wbin, survey, nsample_dropout, add_inputs, test_phase):
        self.directory_main = directory_main
        self.directory_input = directory_input
        self.index1_ini = index1_ini
        self.texp = texp
        self.size_latent_main = size_latent_main
        self.size_latent_ext = size_latent_ext
        self.img_size = img_size
        self.channels = channels
        self.bins = bins
        self.z_min = z_min
        self.z_max = z_max
        self.wbin = wbin
        self.survey = survey
        self.nsample_dropout = nsample_dropout
        self.add_inputs = add_inputs
        self.test_phase = test_phase

        if survey == 1:
            properties, obj_train, obj_test, obj_other = LoadSDSS(directory_main, index1_ini, add_inputs).load_data()

        if survey == 2:
            properties, obj_train, obj_test, obj_other, obj_trainD, obj_trainW = LoadCFHTLS(directory_main, index1_ini, add_inputs, z_max).load_data()

        if survey == 3:
            properties, obj_train, obj_test, obj_other = LoadKIDS(directory_main, index1_ini, add_inputs).load_data()

        self.properties = properties
        self.obj_train = obj_train
        self.obj_test = obj_test
        self.obj_other = obj_other
        print (len(self.obj_train), len(self.obj_test), len(self.obj_other))

          

    def load_img(self, obj):
        if self.survey == 1:
            obj_prix = ('00000' + str(obj))[-6:-4]
            img = np.load(self.directory_main + 'SDSS/sdss_img' + obj_prix + '/' + str(obj) + '.py.npy') 
        
        if self.survey == 2:
            img = np.array(list(self.properties['imgs'][obj]))
    
        if self.survey == 3:
            img = np.load(self.directory_main + 'KiDs_dataset/image_cutout/' + str(obj) + '.npy') 
        return img
    


    def get_output_data(self, obj):         
        if self.test_phase == 1:
            obj_output = obj
        else: 
            obj_output = self.obj_test[:2000]
            
        imgs_output = np.zeros((len(obj_output), self.img_size, self.img_size, self.channels))
        for i in range(len(obj_output)):
            img = self.load_img(obj_output[i])
            imgs_output[i] = img_norm(img)
         
        inputadd_output = self.properties['inputadd'][obj_output]
        z_output = self.properties['z'][obj_output]
        y_output = np.zeros((len(obj_output), self.bins))
        for i in range(len(obj_output)):
            z_index = max(0, min(self.bins - 1, int((z_output[i] - self.z_min) / self.wbin)))
            y_output[i, z_index] = 1.0        
        
        return imgs_output, z_output, y_output, inputadd_output




    def get_zstats(self, zphot_q, z_q, p_out_q, y_q):
        deltaz = (zphot_q - z_q) / (1 + z_q)
        residual = np.mean(deltaz)
        sigma_mad = 1.4826 * np.median(abs(deltaz - np.median(deltaz))) 
        if self.survey == 2 or self.survey == 3: eta_th = 0.15
        elif self.survey == 1: eta_th = 0.05
        eta = len(deltaz[abs(deltaz) > eta_th]) / float(len(deltaz))
        crps = np.mean(np.sum((np.cumsum(p_out_q, 1) - np.cumsum(y_q[:, :self.bins], 1)) ** 2, 1)) * self.wbin                        
        return residual, sigma_mad, eta, crps



    def get_zpoints(self, zlist, p_out_q, num):
        zphot_mean = np.sum(p_out_q * np.expand_dims(zlist, 0), 1)

        zphot_max = np.zeros(num)
        for i in range(num):
            zphot_max[i] = zlist[np.argmax(p_out_q[i])]

        zphot_med = np.zeros(num)
        for i in range(num):
            zphot_med[i] = zlist[np.argmin(abs(np.cumsum(p_out_q[i]) - 0.5))]
        return zphot_mean, zphot_max, zphot_med
                
    
                
    def get_cost_z_indiv(self, p_batch, y_batch):
        return -1 * np.sum(y_batch * np.log(p_batch + 10**(-20)), 1)
    
    

    def get_entropy_indiv(self, p_batch):
        return -1 * np.sum(p_batch * np.log(p_batch + 10**(-20)), 1)



    def get_wpdf_indiv(self, p_batch):
        batch, bins = p_batch.shape
        wpdf_indiv = np.zeros(batch)
        pcdf_axis = np.arange(bins)
        
        pcdf_batch = np.cumsum(p_batch, 1)
        
        for i in range(batch):
            filt_i = (pcdf_batch[i] > 0.16) & (pcdf_batch[i] < 1 - 0.16)
            wpdf_indiv[i] = len(pcdf_axis[filt_i]) * self.wbin
        return wpdf_indiv
    


    def get_cost_z_stats(self, data_q, session, x, y, inputadd, x2, y2, inputadd2, p1, cost1, latent1):

        if self.test_phase == 1:
            obj_all = np.concatenate([self.obj_test, self.obj_train])
            num = len(obj_all)            
        else:
            imgs_q, z_q, y_q, inputadd_q = data_q
            num = len(z_q)
                
        batch = 128
        N_set = len(p1)  # 2    # [p11, p12]
        p_out_q = np.zeros((N_set, num, self.bins))
        cost_z_q = np.zeros(N_set)

        if self.test_phase == 1:
            cost_z_indiv_q = np.zeros((N_set, num))
            entropy_indiv_q = np.zeros((N_set, num))
            wpdf_indiv_q = np.zeros((N_set, num))
            
            if self.size_latent_main <= 16 and self.texp == 10:
                latent_size = N_set * self.size_latent_main
            else:
                latent_size = self.size_latent_main
         #   latent_q = np.zeros((num, self.size_latent_main + self.size_latent_ext)) 
            latent_q = np.zeros((num, latent_size))
            
            
        for i in range(0, num, batch):
            index_i = np.arange(i, min(i + batch, num))
            if self.test_phase == 1:
                imgs_batch, z_batch, y_batch, inputadd_batch = self.get_output_data(obj_all[index_i])
            else:
                imgs_batch = imgs_q[index_i]                
                y_batch = y_q[index_i]
                inputadd_batch = inputadd_q[index_i]
                            
            imgs_batch2 = np.concatenate([imgs_batch[int(batch/2):], imgs_batch[:int(batch/2)]])
            inputadd_batch2 = np.concatenate([inputadd_batch[int(batch/2):], inputadd_batch[:int(batch/2)]])
            y_batch2 = np.concatenate([y_batch[int(batch/2):], y_batch[:int(batch/2)]])
            feed_dict = {x:imgs_batch, y:y_batch, inputadd:inputadd_batch, x2:imgs_batch2, y2:y_batch2, inputadd2:inputadd_batch2}

            output_batch = session.run(p1 + cost1 + latent1, feed_dict = feed_dict)
            for j in range(N_set):
                p_out_q[j][index_i] = output_batch[j]
                cost_z_q[j] = cost_z_q[j] + output_batch[N_set+j] * len(imgs_batch)
                if self.test_phase == 1:
                    cost_z_indiv_q[j][index_i] = self.get_cost_z_indiv(output_batch[j], y_batch)
                    entropy_indiv_q[j][index_i] = self.get_entropy_indiv(output_batch[j])
                    wpdf_indiv_q[j][index_i] = self.get_wpdf_indiv(output_batch[j])
                
            if self.test_phase == 1:
                latent_q[index_i] = output_batch[-1]
                                
        zlist = (0.5 + np.arange(self.bins)) * self.wbin
        cost_z_q = cost_z_q / num
        if self.test_phase == 1:
            zphot_mean = np.zeros((N_set, num))
            zphot_max = np.zeros((N_set, num))
            zphot_med = np.zeros((N_set, num))
            for j in range(N_set):
                zphot_mean_j, zphot_max_j, zphot_med_j = self.get_zpoints(zlist, p_out_q[j], num)
                zphot_mean[j] = zphot_mean_j
                zphot_max[j] = zphot_max_j
                zphot_med[j] = zphot_med_j
            return cost_z_q, latent_q, p_out_q[0], zphot_mean, zphot_max, zphot_med, cost_z_indiv_q, entropy_indiv_q, wpdf_indiv_q
        else:
            residual = np.zeros(N_set)
            sigma_mad = np.zeros(N_set)
            eta = np.zeros(N_set)
            crps = np.zeros(N_set)
            for j in range(N_set):
                zphot_mean = np.sum(p_out_q[j] * np.expand_dims(zlist, 0), 1)
                residual_j, sigma_mad_j, eta_j, crps_j = self.get_zstats(zphot_mean, z_q, p_out_q[j], y_q)
                residual[j] = residual_j
                sigma_mad[j] = sigma_mad_j
                eta[j] = eta_j
                crps[j] = crps_j     
            return cost_z_q, residual, sigma_mad, eta, crps



    def get_avg_hom(self, p_set):
        n = len(p_set)
        s = np.sum(np.array([1 / p_set[i] for i in range(n)]), 0)
        ph = n / s
        ph[np.isnan(ph)] = 0
        ph = ph / np.sum(ph, 1, keepdims=True)
        return ph
    
    

    ### test with dropout
    def get_cost_z_stats_dropout(self, data_q, session, x, y, inputadd, x2, y2, inputadd2, p1, cost1, latent1):

        obj_all = np.concatenate([self.obj_test, self.obj_train])
        num = len(obj_all)
                
        batch = 128
        p_out_q = np.zeros((num, self.bins))
        cost_z_q = 0

        cost_z_indiv_q = np.zeros(num)
        entropy_indiv_q = np.zeros(num)
        wpdf_indiv_q = np.zeros(num)
        latent_q = 0
            
            
        for i in range(0, num, batch):
            index_i = np.arange(i, min(i + batch, num))
            imgs_batch, z_batch, y_batch, inputadd_batch = self.get_output_data(obj_all[index_i])
   
            imgs_batch2 = np.concatenate([imgs_batch[int(batch/2):], imgs_batch[:int(batch/2)]])
            inputadd_batch2 = np.concatenate([inputadd_batch[int(batch/2):], inputadd_batch[:int(batch/2)]])
            y_batch2 = np.concatenate([y_batch[int(batch/2):], y_batch[:int(batch/2)]])
            feed_dict = {x:imgs_batch, y:y_batch, inputadd:inputadd_batch, x2:imgs_batch2, y2:y_batch2, inputadd2:inputadd_batch2}

            poutput_batch = np.zeros((self.nsample_dropout, len(index_i), self.bins))
            for k in range(self.nsample_dropout):
                poutput_batch[k] = session.run(p1[0], feed_dict = feed_dict)
            
            p_out_q[index_i] = self.get_avg_hom(poutput_batch)
                
            cost_z_indiv_q[index_i] = self.get_cost_z_indiv(p_out_q[index_i], y_batch)
            entropy_indiv_q[index_i] = self.get_entropy_indiv(p_out_q[index_i])
            wpdf_indiv_q[index_i] = self.get_wpdf_indiv(p_out_q[index_i])
                                
        zlist = (0.5 + np.arange(self.bins)) * self.wbin
        zphot_mean, zphot_max, zphot_med = self.get_zpoints(zlist, p_out_q, num)
         
        return cost_z_q, latent_q, p_out_q, zphot_mean, zphot_max, zphot_med, cost_z_indiv_q, entropy_indiv_q, wpdf_indiv_q




 
    def get_obj_nonoverlap(self, obj_all, obj_pre, subbatch):
        obj_select = np.random.choice(obj_all, subbatch)
        for i in range(subbatch):
            while obj_select[i] == obj_pre[i]:
                obj_select[i] = np.random.choice(obj_all)
        return obj_select


    def get_obj_nonoverlap_z(self, obj_all, obj_pre, subbatch):
        obj_select = np.random.choice(obj_all, subbatch)
        for i in range(subbatch):
            z_index_pre = max(0, min(self.bins - 1, int((self.properties['z'][obj_pre[i]] - self.z_min) / self.wbin)))
            z_index_select = max(0, min(self.bins - 1, int((self.properties['z'][obj_select[i]] - self.z_min) / self.wbin)))

            while z_index_select == z_index_pre:
                obj_select[i] = np.random.choice(obj_all)
                z_index_select = max(0, min(self.bins - 1, int((self.properties['z'][obj_select[i]] - self.z_min) / self.wbin)))

        return obj_select
    
    

    
        
    def get_next_subbatch(self, subbatch):

        if self.texp == 0:
            obj1_subbatch = np.random.choice(self.obj_train, subbatch)
            obj_list = [obj1_subbatch]
            get_aug = [False]

        if self.texp == 10:
            obj1_subbatch = np.random.choice(self.obj_train, subbatch)
            obj2_subbatch = self.get_obj_nonoverlap(self.obj_train, obj1_subbatch, subbatch) 
            obj_list = [obj1_subbatch, obj2_subbatch]
            get_aug = [True, True]
            
            
        x_list = []
        y_list = []
        inputadd_list = []
        
        for k, obj_subbatch in enumerate(obj_list):        
            obj_subbatch = np.array(obj_subbatch)
            z = self.properties['z'][obj_subbatch]
            inputadd = self.properties['inputadd'][obj_subbatch]
            
            y = np.zeros((subbatch, self.bins))
            x = np.zeros((subbatch, self.img_size, self.img_size, self.channels))
            
            x_morph = np.zeros((subbatch, self.img_size, self.img_size, self.channels))
                 
            for i in range(subbatch):
                img = self.load_img(obj_subbatch[i])               
                img = img_reshape(img)
                x[i] = img_norm(np.array(list(img)))
                    
                z_index = max(0, min(self.bins - 1, int((z[i] - self.z_min) / self.wbin)))
                y[i, z_index] = 1.0                
   
                if get_aug[k]:
                    img_morph = img_morph_aug(np.array(list(img)))
                    x_morph[i] = img_norm(img_morph)

            x = [x, x_morph]
   
            x_list.append(x)
            y_list.append(y)
            inputadd_list.append(inputadd)
                
        return x_list, y_list, inputadd_list
    
    


      