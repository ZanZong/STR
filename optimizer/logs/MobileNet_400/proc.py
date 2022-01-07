import numpy as np
def gen_appro_checkmate_r():
    origin = "/home/zongzan/dist_dnn_training/STR/optimizer/logs/MobileNet_400/R-checkmate-appro"
    r_appro = np.loadtxt(origin)
    new_r = np.zeros_like(r_appro, dtype=int)
    print(r_appro.shape)
    cout = 0
    for i in range(r_appro.shape[0]):
        for j in range(r_appro.shape[1]):
            if i == j:
                new_r[i, j] = 1
            if r_appro[i, j] == 1.0:
                new_r[i, j] = 1
                cout += 1
    print(cout)

    np.savetxt("/home/zongzan/dist_dnn_training/STR/optimizer/logs/MobileNet_400/R-checkmate", new_r, fmt="%i")

def fix_str_p():
    origin = "/home/zongzan/dist_dnn_training/STR/optimizer/logs/vgg_unet_40/Q-approx"
    q_appro = np.loadtxt(origin)
    new_r = np.zeros_like(q_appro, dtype=int)
    for i in range(q_appro.shape[0]):
        for j in range(q_appro.shape[1]):
            if q_appro[i, j] == 1:
                new_r[j + 1, j] = 1
    
    np.savetxt("/home/zongzan/dist_dnn_training/STR/optimizer/logs/vgg_unet_40/P-approx", new_r, fmt="%i")

# gen_appro_checkmate_r()
fix_str_p()