import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import time
import threading
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--survey', help='survey', type=int)
parser.add_argument('--recali_with_nval', help='recali_with_nval', type=int)
parser.add_argument('--ne', help='No. Expriment', type=int)
parser.add_argument('--test_phase', help='if conducting the test phase', type=int)

args = parser.parse_args()
survey = args.survey
recali_with_nval = args.recali_with_nval
ne = args.ne
test_phase = args.test_phase


survey3_onlyebv = False
save_all = True
count_time = False



####################
##### Settings #####
####################



directory_main = '/userhome/'
directory_io = directory_main + 'PZlatent/'


##### Iterations, learning rate, training batch sizes, data, expriment settings #####


iterations = 40000  #80000  #20000
   
ite_point_save = [iterations]  # save model at number of iterations during training
learning_rate_step = 100000000000  #40000  #iterations / 2


learning_rate_ini = 0.0001
lr_reduce_factor = 5.0
ite_test_print = 2000  # print test results per number of iterations during training
num_test_est = 2000    


batch_train = 128


cross_val = 1
num_expri = 1
repetition_per_ite = 1
use_cpu = False
clip_grad = False

num_threads = 4 + 1
        

if survey == 1:
    
    bins = 180
    z_min = 0.0
    z_max = 0.4
    z_prix = 'z04_bin' + str(bins)
    
    n_test_ext = 103305 + 20000
    n_test = 103305
    n_train = 393219
    
    obj_train = np.load(directory_main + 'SDSS/train.npy')
    obj_val = np.load(directory_main + 'SDSS/val.npy')
    obj_test = np.load(directory_main + 'SDSS/test.npy')
    
    err_index = np.argwhere(obj_test == 281531)
    obj_test = np.delete(obj_test, err_index) 
    
    obj_test = np.concatenate([obj_test, obj_val])
    
    metadata = np.load(directory_main + 'SDSS/label.npz')
    z = metadata['z']
    z_test = z[obj_test]


    size_latent = 16
    
    ite = 150000
    sprix = 'S' + str(ite)
    datalabel = 'latent1a1ei_S' + str(ite) + '_'
    fpre = 'probas_net111_ADAM5_texp10_batch32_ite180000_n16s512_ebvImg_addLayer1_noAddInput_trainAll_imgnorm5_zContra_trainS2_scratch_z04_bin180_cv1ne1__iteAlter'
    
     
        
if survey == 2:
    
    bins = 1000
    z_min = 0.0
    z_max = 4.0
    z_prix = 'z4_bin' + str(bins)
    
    n_test_ext = 34759  #20000
    n_test = 20000
    n_train = 100000   #123434  #138193
    
    obj_train = np.load(directory_main + 'CFHT/trainc2.npy')[:n_train]
    obj_test = np.load(directory_main + 'CFHT/testc2.npy')
    
    cfht_cat = np.load(directory_main + 'CFHT/CFHTLS_catalog.npz')
    z = cfht_cat['zspec']
    z_test = z[obj_test]
    
      
    size_latent = 16
    
    ite = 120000
    sprix = 'C_15rand' + str(ite)
    datalabel = 'latent1a1ei_C_15rand_onlyHQ_' + str(ite) + '_'
    fpre = 'probas_net111_ADAM5_texp10_batch32_ite150000_n16s512_ebvImg_addLayer1_noAddInput_trainAll_coeffRecon1p0_imgnorm6_zContra_trainCmixLR2_randWD_scratch_z4_bin1000_cv1ne1__iteAlter'
    
    


if survey == 3:
    
    bins = 800
    z_min = 0.0
    z_max = 3.0
    z_prix = 'z3_bin' + str(bins)
    
    n_test_ext = 34147
    n_test = 20000
    n_train = 100000  
    
    obj_train = np.load(directory_main + 'KiDs_dataset/traink.npy')
    obj_test = np.load(directory_main + 'KiDs_dataset/testk.npy')
    
    metadata = np.load(directory_main + 'KiDs_dataset/catalog_kids.npz')
    z = metadata['zspec']
    z_test = z[obj_test]
         
         
    size_latent = 16
            
    ite = 120000
    
    if survey3_onlyebv:
        sprix = 'Konlyebv_15rand' + str(ite)
        datalabel = 'latent1a1ei_K_15rand_' + str(ite) + '_'
        fpre = 'probas_net111_ADAM5_texp10_batch32_ite150000_n16s512_ebvImg_addLayer1_noAddInput_trainAll_imgnorm5_zContra_trainKIDS2rand_scratch_z3_bin800_cv1ne1__iteAlter'
    else:
        sprix = 'K_15rand' + str(ite)
        datalabel = 'latent1a6ei_K_15rand_' + str(ite) + '_'
        fpre = 'probas_net111_ADAM5_texp10_batch32_ite150000_n16s512_ebvmagImg_addLayer1_noAddInput_trainAll_imgnorm5_zContra_trainKIDS2rand_scratch_z3_bin800_cv1ne1__iteAlter'
    

if ne > 1:
    datalabel = datalabel.replace('ei', 'ei' + str(ne) + 'ne')
    fpre = fpre.replace('cv1ne1', 'cv1ne' + str(ne))

if recali_with_nval == -1:
    fpdf = directory_io + 'resne_collect_' + sprix + '_raw_uncali.npz'
if recali_with_nval == 0:
    fpdf = directory_io + 'resne_collect_' + sprix + '_raw.npz'
if recali_with_nval == 1:
    fpdf = directory_io + 'resne_collect_' + sprix + '_raw_nval.npz'
if recali_with_nval == 2:
    fpdf = directory_io + 'resne_collect_' + sprix + '_raw_trainval.npz'
    
f = np.load(directory_io + fpre + str(ite) + '_.npz')      

ne_point = np.array([1, 3, 4, 5, 6, 7, 8, 9, 10, 11])
k_ne = np.argwhere(ne_point == ne)[0][0]

label = np.load(fpdf)['pdf_knn'][k_ne]
print (k_ne)
    

latent = f['latent'][:, :size_latent][:n_test_ext+n_train]
latent_test = latent[:n_test_ext]

label_test = label[:n_test_ext]
    

wbin = (z_max - z_min) / bins
zlist = (0.5 + np.arange(bins)) * wbin


 
filt = np.sum(label_test, 1) == 0
label_test = label_test / np.sum(label_test, 1, keepdims=True)
label_test[filt] = 1.0 / bins
print ('norm')
print (label_test.shape, np.min(np.sum(label_test, 1)), np.max(np.sum(label_test, 1)), label_test[filt].shape)
   

    

if survey == 1:
    num_train = int(n_test / 2)
    latent_train = latent_test[:num_train]
    label_train = label_test[:num_train]
else:   
    num_train = n_test
    latent_train = latent_test[:n_test]
    label_train = label_test[:n_test]
    

    



###### Model load / save paths #####

TrainData = datalabel + 'texp10_'



if recali_with_nval == -1:
    Algorithm = 'fit_uncal_'
elif recali_with_nval == 0:
    Algorithm = 'fit_'
elif recali_with_nval == 1:
    Algorithm = 'fit_nval_'
elif recali_with_nval == 2:
    Algorithm = 'fit_trainval_'
elif recali_with_nval == 3:
    Algorithm = 'fit_wtrainval_'

Algorithm = 'pre' + Algorithm + 'ADAM_wasser2ce_'

if repetition_per_ite != 1: Algorithm = Algorithm + 'rep' + str(repetition_per_ite) + '_'
Algorithm = Algorithm + 'batch' + str(batch_train) + '_'
Algorithm = Algorithm + 'ite' + str(iterations) + '_'

Pretrain = 'scratch_'

fx = Algorithm + TrainData + Pretrain + z_prix + '_'
    
model_savepath = directory_io + 'model_' + fx + '/'
z_savepath = directory_io + 'probas_' + fx

    
fi = open(directory_io + 'f_' + fx + '.txt', 'a')
fi.write(fx + '\n\n')
fi.close()

print ('#####') 
print (fx)
print ('#####') 



if test_phase == 1:
    if save_all:
        latent_save = latent
        z_savepath = z_savepath + 'saveAll_'
    else:
        latent_save = latent_test
    


##### Network & Cost Function #####


def prelu(x):
    with tf.name_scope('PRELU'):
        _alpha = tf.get_variable('prelu', shape=x.get_shape()[-1], dtype = x.dtype, initializer=tf.constant_initializer(0.0))
    return tf.maximum(0.0, x) + _alpha * tf.minimum(0.0, x)

   
def fully_connected(input, num_outputs, name, act='relu', reuse=False):           
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        num_input_units = input.get_shape()[-1].value
        weights_shape = [num_input_units, num_outputs]
    #    weights = tf.get_variable('weights', shape=weights_shape, initializer=tf.contrib.layers.xavier_initializer())
        weights = tf.get_variable('weights', shape=weights_shape, initializer=tf.glorot_uniform_initializer())
        biases = tf.get_variable('biases', shape=[num_outputs], initializer=tf.constant_initializer(0.1))
        
        outputs = tf.matmul(input, weights)
        outputs = tf.nn.bias_add(outputs, biases)

        if act == 'prelu': outputs = prelu(outputs)
        elif act == 'relu': outputs = tf.nn.relu(outputs)
        elif act == 'tanh': outputs = tf.nn.tanh(outputs)
        elif act == 'sigmoid': outputs = tf.sigmoid(outputs)
        elif act == 'leakyrelu': outputs = tf.nn.leaky_relu(outputs)
        elif act == None: pass
        return outputs


                
    
lr = tf.placeholder(tf.float32, shape=[], name='lr')
x = tf.placeholder(tf.float32, shape=[None, size_latent], name='x')
y = tf.placeholder(tf.float32, shape=[None, bins], name='y')


fc1 = fully_connected(input=x, num_outputs=1024, name='fc1', act='relu')
output = fully_connected(input=fc1, num_outputs=bins, name='fc2', act=None)
p = tf.nn.softmax(output)

cost1 = tf.reduce_mean(abs(tf.cumsum(y, 1) - tf.cumsum(p, 1)) * z_max, 1)
cost2 = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
     
cost = tf.reduce_mean(100 * cost1 * cost2)
entropy = -1 * tf.reduce_mean(y * tf.log(y+10**(-20)))
    
    

##### Session, saver or optimizer #####


if use_cpu: session_conf = tf.ConfigProto(device_count={'GPU':0})#log_device_placement=True
else:
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=session_conf)

tvars = tf.trainable_variables()
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
if clip_grad:
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=5.0)
optimizer = optimizer.minimize(cost, var_list=tvars)      
session.run(tf.global_variables_initializer())
        

if test_phase == 1:
    iterations = 0
    tvars = tf.trainable_variables()
    saver = tf.train.Saver(var_list=tvars)
    saver.restore(session, tf.train.latest_checkpoint(model_savepath))
    
    


def get_zstats(zphot_q, z_q, p_out_q, y_q):
    deltaz = (zphot_q - z_q) / (1 + z_q)
    residual = np.mean(deltaz)
    sigma_mad = 1.4826 * np.median(abs(deltaz - np.median(deltaz)))    
    eta_th = 0.05
    eta = len(deltaz[abs(deltaz) > eta_th]) / float(len(deltaz))
    crps = np.mean(np.sum((np.cumsum(p_out_q, 1) - np.cumsum(y_q, 1)) ** 2, 1)) * wbin
    return residual, sigma_mad, eta, crps


def get_zpoints(p_out_q):
    num = len(p_out_q)
    zphot_mean = np.sum(p_out_q * np.expand_dims(zlist, 0), 1)

    zphot_max = np.zeros(num)
    for i in range(num):
        zphot_max[i] = zlist[np.argmax(p_out_q[i])]

    zphot_med = np.zeros(num)
    for i in range(num):
        zphot_med[i] = zlist[np.argmin(abs(np.cumsum(p_out_q[i]) - 0.5))]
    return zphot_mean, zphot_max, zphot_med
                
                


def get_cost_z_stats(data_q, session, x, y, pred):

    x_q, z_q, y_q = data_q
    batch = 1000
    num = len(x_q)
    p_out_q = np.zeros((num, bins))
    cost_q = 0
    entropy_q = 0
    
    for i in range(0, num, batch):
        index_i = np.arange(i, min(i + batch, num))
        x_batch = x_q[index_i]                
        y_batch = y_q[index_i]
                            
        feed_dict = {x:x_batch, y:y_batch}

        p_out_q_i, cost_q_i, entropy_q_i = session.run(pred, feed_dict = feed_dict)
        p_out_q[index_i] = p_out_q_i
        cost_q = cost_q + cost_q_i * len(x_batch)
        entropy_q = entropy_q + entropy_q_i * len(x_batch)
    cost_q = cost_q / num
    entropy_q = entropy_q / num
    
    zphot_mean, zphot_max, zphot_med = get_zpoints(p_out_q)
    residual, sigma_mad, eta, crps = get_zstats(zphot_mean, z_q, p_out_q, y_q)                      
    return cost_q, entropy_q, p_out_q, zphot_mean, zphot_max, zphot_med, residual, sigma_mad, eta, crps



def get_z_stats_simple(x_q, session, x, pred):

    batch = 1000
    num = len(x_q)
    p_out_q = np.zeros((num, bins))
    
    start = time.time()
    for i in range(0, num, batch):
        index_i = np.arange(i, min(i + batch, num))
        x_batch = x_q[index_i]                
                            
        feed_dict = {x:x_batch}

        p_out_q_i = session.run(pred, feed_dict = feed_dict)
        p_out_q[index_i] = p_out_q_i

        if count_time and (i+batch) % 2000 == 0: 
            print(i+batch, str((time.time() - start) / 60) + ' minutes')
    
    zphot_mean, zphot_max, zphot_med = get_zpoints(p_out_q)
    return p_out_q, zphot_mean, zphot_max, zphot_med





##### Training #####


def Train(i, th, fi):
    global x_, y_
    global running
    
    if th == 0:
        feed_dict = {x:x_, y:y_, lr:learning_rate}
            
        if i == 0 or (i + 1) % ite_test_print == 0:
            print ('iteration:' + str(i+1) + ' lr:' + str(learning_rate) + ' batch:' + str(batch_train) + ' time:' + str((time.time() - start) / 60) + ' minutes')

            fi = open(directory_io + 'f_' + fx + '.txt', 'a')
            fi.write('iteration:' + str(i+1) + ' lr:' + str(learning_rate) + ' batch:' + str(batch_train) + ' time:' + str((time.time() - start) / 60) + ' minutes' + '\n')
            fi.write('cost_training: ' + str(session.run([cost, entropy], feed_dict = feed_dict)) + '\n')
            
            cost_q, entropy_q, _, _, _, _, residual, sigma_mad, eta, crps = get_cost_z_stats([latent_test[:num_test_est], z_test[:num_test_est], label_test[:num_test_est]], session, x, y, [p, cost, entropy])
            fi.write('outputs_test: ' + str([cost_q, entropy_q, residual, sigma_mad, eta, crps]) + '\n\n')
            fi.close()
            

        for t in range(repetition_per_ite):
            session.run(optimizer, feed_dict = feed_dict)
        running = 0
        
    else:
        def read_data(j):
            index_j = np.arange((j-1)*int(batch_train/(num_threads-1)), j*int(batch_train/(num_threads-1)))
            subbatch = len(index_j)
            obj_sample = np.random.choice(num_train, subbatch)

            while True:
                if running == 0: break
            
            x_[index_j] = latent_train[obj_sample]
            y_[index_j] = label_train[obj_sample]
                                    
        for j in range(1, num_threads):
            if th == j:                
                read_data(j)


if test_phase == 0:                  
    obj_sample = np.random.choice(num_train, batch_train)
    x_ = latent_train[obj_sample]
    y_ = label_train[obj_sample]
            
    
    learning_rate = learning_rate_ini
    start = time.time()
    print ('Start training...')
           
    for i in range(iterations):
        if (i + 1) % learning_rate_step == 0: learning_rate = learning_rate / lr_reduce_factor
        running = 1
            
        threads = []
        for th in range(num_threads):
            t = threading.Thread(target = Train, args = (i, th, fi))
            threads.append(t)
        for th in range(num_threads):
            threads[th].start()
        for th in range(num_threads):
            threads[th].join()    
            
        if (i + 1) in ite_point_save:
            saver = tf.train.Saver()
            saver.save(session, model_savepath, i)
            

##### Saving #####

if test_phase == 1:
    p_out_q, zphot_mean, zphot_max, zphot_med = get_z_stats_simple(latent, session, x, p)
    
    np.savez(z_savepath, zphot_mean=zphot_mean, zphot_max=zphot_max, zphot_med=zphot_med, p_out=p_out_q)
    
    deltaz = (zphot_mean[:n_test_ext] - z_test) / (1 + z_test)
    residual = np.mean(deltaz)
    sigma_mad = 1.4826 * np.median(abs(deltaz - np.median(deltaz))) 
    print (residual, sigma_mad)

print (fx)
