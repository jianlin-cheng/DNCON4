import sys

#source /storage/htc/bdm/Collaboration/Zhiye/Vir_env/DNCON4_vir_cpu/bin/activate
#python /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/lib/dist2image.py evadir/pred_map/3BFO-B-distance.txt  evadir/pred_map/3BFO-B-distance.png

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
  print('please input the right parameters')
  sys.exit(1)

dist_file=sys.argv[1] #
outimage=sys.argv[2] #


cmap = np.loadtxt(dist_file,dtype='float32')
L = cmap.shape[0]
Y = np.zeros((L, L))
value=L
i = 0
with open(dist_file) as f:
    for line in f:
        if line.startswith('#'):
            continue
        this_line = line.strip().split()
        if len(this_line) != value:
            print("\nThis_line = %i, L = %i, Lable file %s error!\n" % (len(this_line), value, dist_file))
            Y = [0]
        Y[i, 0:L] = np.asarray(this_line)
        i = i + 1

# all_Y.append(Y)
plt.imshow(Y)
plt.colorbar()
plt.savefig(outimage)
plt.close()
