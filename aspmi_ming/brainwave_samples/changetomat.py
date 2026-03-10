import numpy as np
import scipy.io as sio
import os

# 使用脚本所在目录，这样在 brainwave_samples 内或父目录运行都可以
folder = os.path.dirname(os.path.abspath(__file__))

for f in os.listdir(folder):
    if f.endswith(".npy"):
        path_npy = os.path.join(folder, f)
        path_mat = os.path.join(folder, f.replace(".npy", ".mat"))
        data = np.load(path_npy)
        sio.savemat(path_mat, {"data": data})
        print(f"Converted: {f} -> {f.replace('.npy', '.mat')}")
