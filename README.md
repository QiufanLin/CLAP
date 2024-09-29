# CLAP – I: Resolving miscalibration for deep learning-based galaxy photometric redshift estimation

(TO BE COMPLETED)


# Training and Testing

The code is tested using: 
- Python 2.7.15
- TensorFlow 1.12.0
- CPU: Intel(R) Core(TM) i9-7920X
- GPU: Titan V / GeForce RTX 2080 Ti


## Supervised contrastive learning

*** Both training and testing have to be run consecutively; this applies to all the following cases.

*** Run different folds of experiments by setting "--ne" = 1,2,3,4,5.

- (training):
> python CLAP_main.py --ne=1 --survey=1 --bins=180 --net=111 --size_latent_main=16 --size_latent_ext=512 --batch_train=32 --texp=10 --add_inputs=1 --itealter=0 —test_phase=0

- (testing):
Set "--test_phase=1"
> python CLAP_main.py --ne=1 --survey=1 --bins=180 --net=111 --size_latent_main=16 --size_latent_ext=512 --batch_train=32 --texp=10 --add_inputs=1 --itealter=0 —test_phase=1
