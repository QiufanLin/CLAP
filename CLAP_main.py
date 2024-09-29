import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import time
from CLAP_network import *
from CLAP_data import *
import threading
import argparse



####################
##### Settings #####
####################



directory_main = '/userhome/'  # '/renoir_data_02/qiufan/'
directory_input = directory_main + 'PZlatent/'
directory_output = directory_main + 'PZlatent/'
index1_ini = np.load(directory_main + 'random_seed_1000000.npy')

parser = argparse.ArgumentParser()
parser.add_argument('--ne', help='# experiment', type=int)
parser.add_argument('--test_phase', help='if conducting the test phase', type=int)
parser.add_argument('--survey', help='select a survey', type=int)
parser.add_argument('--bins', help='bins', type=int)
parser.add_argument('--net', help='select a network', type=int)
parser.add_argument('--size_latent_main', help='size of the main latent space', type=int)
parser.add_argument('--size_latent_ext', help='size of the extended latent space', type=int)
parser.add_argument('--batch_train', help='size of a mini-batch', type=int)
parser.add_argument('--texp', help='experiment type', type=int)
parser.add_argument('--add_inputs', help='add additional inputs as images, including EBV', type=int)
parser.add_argument('--itealter', help='the altered checkpoint (iteration)', type=int)


args = parser.parse_args()
num_expri = args.ne
test_phase = args.test_phase
survey = args.survey
bins = args.bins
net = args.net
size_latent_main = args.size_latent_main
size_latent_ext = args.size_latent_ext
batch_train = args.batch_train
texp = args.texp
add_inputs = args.add_inputs
itealter = args.itealter



coeff_recon = 100
imgnorm = 5
if survey == 2:  # CFHTLS
    coeff_recon = 1
    imgnorm = 6

add_latent_prelayer = 1

num_gmm = 0  #5

ratio_dropout = 0  #0.5
nsample_dropout = 100




##### Iterations, learning rate, training batch sizes, data, expriment settings #####


#iterations = 180000  # 20000     # total number of iterations for training

if survey == 2 or survey == 3:  # CFHTLS, KiDS
    iterations = 150000
    ite_point_save = [i for i in range(90000, 150000+6000, 6000)]
if survey == 1:                 # SDSS
    iterations = 180000
    ite_point_save = [i for i in range(120000, 180000+6000, 6000)]
    
learning_rate_step = 60000  #iterations / 2


learning_rate_ini = 0.0001
lr_reduce_factor = 5.0
ite_test_print = 2000   #200  # print test results per number of iterations during training
    
    
img_size = 64
channels = 5
if survey == 3:
    channels = 4

cross_val = 1
repetition_per_ite = 1
use_cpu = False
clip_grad = False
num_threads = int(batch_train/2) + 1
#num_threads = batch_train + 1


alter_checkpoint = False
if itealter != 0: 
    alter_checkpoint = True
    ite_altercheckpoint = itealter
    
    


###### Redshift #####
    
    
if survey == 1:  # SDSS
    z_min = 0.0
    z_max = 0.4
    z_prix = 'z04_bin' + str(bins)

elif survey == 2:  # CFHTLS
    z_min = 0.0
    z_max = 4.0
    z_prix = 'z4_bin' + str(bins)

if survey == 3:  # KIDS
    z_min = 0.0
    z_max = 3.0
    z_prix = 'z3_bin' + str(bins) 
    
wbin = (z_max - z_min) / bins
z_prix = z_prix + '_cv' + str(cross_val) + 'ne' + str(num_expri)
        
     

###### Model load / save paths #####


if net == 0: Net = 'netP_'
elif net >= 10: Net = 'net' + str(net) + '_'
elif net > 1: Net = 'net' + str(net - 1) + '_'
 

if survey == 1: 
    TrainData = 'trainS2_'  
elif survey == 2:
    TrainData = 'trainCmixLR2_randWD_'
elif survey == 3: 
  #  TrainData = 'trainKIDS2_'
    TrainData = 'trainKIDS2rand_'
    

Algorithm = 'ADAM5_texp' + str(texp) + '_'


if repetition_per_ite != 1: Algorithm = Algorithm + 'rep' + str(repetition_per_ite) + '_'
Algorithm = Algorithm + 'batch' + str(batch_train) + '_'
Algorithm = Algorithm + 'ite' + str(iterations) + '_'
Algorithm = Algorithm + 'n' + str(size_latent_main) + 's' + str(size_latent_ext) + '_'

if num_gmm > 0:
    Algorithm = Algorithm + 'GMM' + str(num_gmm) + '_'

if ratio_dropout > 0:
    Algorithm = Algorithm + 'dropout' + str(ratio_dropout).replace('.', '') + 'n' + str(nsample_dropout) + '_'
    

if add_inputs == 1:
    Algorithm = Algorithm + 'ebvImg_'
elif add_inputs == 2:
    Algorithm = Algorithm + 'ebvmagImg_'
elif add_inputs == 3:
    Algorithm = Algorithm + 'ebvmagallImg_'

    
if add_latent_prelayer > 0:
    Algorithm = Algorithm + 'addLayer' + str(add_latent_prelayer) + '_'


Algorithm = Algorithm + 'noAddInput_'
Algorithm = Algorithm + 'trainAll_'


    
if texp == 10 and coeff_recon != 100:                  
    Algorithm = Algorithm + 'coeffRecon' + str(coeff_recon).replace('.', 'p') + '_'
    
Algorithm = Algorithm + 'imgnorm' + str(imgnorm) + '_'

 
if texp == 10:
    Algorithm = Algorithm + 'zContra_'

    
Pretrain = 'scratch_'
    
    
fx = Net + Algorithm + TrainData + Pretrain + z_prix + '_'

model_savepath = directory_output + 'model_' + fx + '/'
z_savepath = directory_output + 'probas_' + fx

if test_phase == 0:
    fi = open(directory_output + 'f_' + fx + '.txt', 'a')
    fi.write(fx + '\n\n')
    fi.close()

if alter_checkpoint and test_phase == 1:
     z_savepath = z_savepath + '_iteAlter' + str(ite_altercheckpoint) + '_'
          
        
     
print ('#####') 
print (fx)
print ('#####') 


if test_phase == 1:
    model_load = directory_input + 'model_' + fx + '/'
    iterations = 0


    
    

##### Load data #####

getdata = GetData(directory_main=directory_main, directory_input=directory_input, index1_ini=index1_ini, 
                  texp=texp, size_latent_main=size_latent_main, size_latent_ext=size_latent_ext, img_size=img_size, 
                  channels=channels, bins=bins, z_min=z_min, z_max=z_max, wbin=wbin, survey=survey, 
                  nsample_dropout=nsample_dropout, add_inputs=add_inputs, test_phase=test_phase)

if test_phase == 0:
    imgs_test, z_test, y_test, inputadd_test = getdata.get_output_data(obj=[])



##### Network & Cost Function #####

   
model = Model(texp=texp, img_size=img_size, channels=channels, bins=bins,
              size_latent_main=size_latent_main, size_latent_ext=size_latent_ext, net=net, num_gmm=num_gmm, ratio_dropout=ratio_dropout, nsample_dropout=nsample_dropout, 
              add_inputs=add_inputs, add_latent_prelayer=add_latent_prelayer,
              name='model_', reuse=False)

if texp == 0:
    p1, cost_ce_single1, latent1 = model.get_outputs()
    cost = cost_ce_single1[0]
    
elif texp == 10:
    cost_ce, cost_recon, cost_contra, cost_zcontra, p1, cost_ce_single1, latent1 = model.get_outputs()
    cost_recon = coeff_recon * cost_recon
    cost = cost_ce + cost_recon + cost_contra + cost_zcontra
        


lr = model.lr

x = model.x
inputadd = model.inputadd
y = model.y
    
x2 = model.x2
inputadd2 = model.inputadd2
y2 = model.y2

x_morph = model.x_morph
x2_morph = model.x2_morph




##### Session, saver or optimizer #####


if use_cpu: session_conf = tf.ConfigProto(device_count={'GPU':0})#log_device_placement=True
else:
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=session_conf)



if test_phase == 1:
    tvars = tf.trainable_variables()
    if texp == 0:
        tvars = [var for var in tvars if ('decoder' not in var.name)]

    if alter_checkpoint:
        f = open(model_load + 'checkpoint', 'r')
        ss = f.readlines()
        f = open(model_load + 'checkpoint', 'w')
        f.write('model_checkpoint_path: "' + model_load + '-' + str(ite_altercheckpoint-1) + '"\n' + 'all_model_checkpoint_paths: "' + model_load + '-' + str(ite_altercheckpoint-1) + '"')
        f.close()

    saver = tf.train.Saver(var_list=tvars)
    saver.restore(session, tf.train.latest_checkpoint(model_load))
    
    if alter_checkpoint:
        f = open(model_load + 'checkpoint', 'w')
        f.write(ss[0]+ss[1])
        f.close()
        
        
else:
    tvars = tf.trainable_variables()
    
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    if clip_grad:
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=5.0)
         
    optimizer = optimizer.minimize(cost, var_list=tvars)
    session.run(tf.global_variables_initializer())
   
    



##### Training #####


def Train(i, th):
    global x_, y_, inputadd_, x2_, y2_, inputadd2_, x_morph_, x2_morph_
    global running
    
    if th == 0:
        feed_dict = {x:x_, y:y_, inputadd:inputadd_, lr:learning_rate}
        if texp == 10:
            feed_dict.update({x2:x2_, y2:y2_, inputadd2:inputadd2_})
            feed_dict.update({x_morph:x_morph_, x2_morph:x2_morph_})
            
        if i == 0 or (i + 1) % ite_test_print == 0:
            fi = open(directory_output + 'f_' + fx + '.txt', 'a')

            fi.write('iteration:' + str(i+1) + ' lr:' + str(learning_rate) + ' batch:' + str(batch_train) + ' time:' + str((time.time() - start) / 60) + ' minutes' + '\n')

            if texp == 0:
                fi.write('cost_training (ce_single): ' + str(session.run(cost, feed_dict = feed_dict)) + '\n')
   
            if texp == 10:
                fi.write('cost_training (ce_single): ' + str(session.run(cost_ce_single1, feed_dict = feed_dict)) + '\n')
                fi.write('cost_training (ce, recon, contra, zcontra): ' + str(session.run([cost_ce, cost_recon, cost_contra, cost_zcontra], feed_dict = feed_dict)) + '\n')

            outputs_test = getdata.get_cost_z_stats([imgs_test, z_test, y_test, inputadd_test], session, x, y, inputadd, x2, y2, inputadd2, p1, cost_ce_single1, latent1)
            fi.write('outputs_test: ' + str(outputs_test) + '\n\n')
            fi.close()
            
        for t in range(repetition_per_ite):
            session.run(optimizer, feed_dict = feed_dict)
        running = 0
        
    else:
        def read_data(j):
            index_j = np.arange((j-1)*int(batch_train/(num_threads-1)), j*int(batch_train/(num_threads-1)))
            subbatch = len(index_j)
            x_list, y_list, inputadd_list = getdata.get_next_subbatch(subbatch)

            while True:
                if running == 0: break
            
            x_[index_j] = x_list[0][0]
            y_[index_j] = y_list[0]
            inputadd_[index_j] = inputadd_list[0]
            x_morph_[index_j] = x_list[0][1]

            if texp == 10:        
                x2_[index_j] = x_list[1][0]
                y2_[index_j] = y_list[1]
                inputadd2_[index_j] = inputadd_list[1]
                x2_morph_[index_j] = x_list[1][1]
                        
                
        for j in range(1, num_threads):
            if th == j:                
                read_data(j)
                   


if test_phase == 0:
    x_list, y_list, inputadd_list = getdata.get_next_subbatch(batch_train)
    
    x_ = x_list[0][0]
    y_ = y_list[0]
    inputadd_ = inputadd_list[0]
    x_morph_ = x_list[0][1]

    if texp == 10:    
        x2_ = x_list[1][0]
        y2_ = y_list[1]
        inputadd2_ = inputadd_list[1]
        x2_morph_ = x_list[1][1]
        

    
    learning_rate = learning_rate_ini
    start = time.time()
    print ('Start training...')
       
    for i in range(iterations):
            
        if (i + 1) % learning_rate_step == 0: learning_rate = learning_rate / lr_reduce_factor
        running = 1
        
        threads = []
        for th in range(num_threads):
            t = threading.Thread(target = Train, args = (i, th))
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
    
    if ratio_dropout == 0:
        cost_z_q, latent_q, p_out_q, zphot_mean, zphot_max, zphot_med, cost_z_indiv_q, entropy_indiv_q, wpdf_indiv_q = getdata.get_cost_z_stats([], session, x, y, inputadd, x2, y2, inputadd2, p1, cost_ce_single1, latent1)
    else:
        cost_z_q, latent_q, p_out_q, zphot_mean, zphot_max, zphot_med, cost_z_indiv_q, entropy_indiv_q, wpdf_indiv_q = getdata.get_cost_z_stats_dropout([], session, x, y, inputadd, x2, y2, inputadd2, p1, cost_ce_single1, latent1)
   
    
    np.savez(z_savepath, zphot_mean=zphot_mean, zphot_max=zphot_max, zphot_med=zphot_med, cost_z_indiv_q=cost_z_indiv_q, entropy_indiv_q=entropy_indiv_q, wpdf_indiv_q=wpdf_indiv_q,
             cost=cost_z_q, latent=latent_q, p_out=p_out_q)
    
print (fx)
