#Architecture
Different network architecture

1.CNN_arch

For binary class, on dncon2 dataset get topL/5 acc at 69.06%, on deepcov dataset get topL/5 acc at 73.92%.

** run on local
```
cd CNN_arch/scripts/local
sh run_sbatch_gpu_2D_tune.sh
```

** run on lewis
```
cd CNN_arch/scripts/lewis
sh run_sbatch_gpu.sh
```

The location of main code entery CNN_arch/scripts/train_deepCNN_2D_gen_tune.py


2.ResNet_arch

For binary class, on dncon2 dataset get topL/5 acc at 65.64%, on deepcov dataset get topL/5 acc at 75.04%.

** run on local
```
cd ResNet_arch/scripts/local
sh run_sbatch_gpu_2D_tune.sh
```

** run on lewis
```
cd ResNet_arch/scripts/lewis
sh run_sbatch_gpu_2D_tune.sh
```

3.UNet_arch