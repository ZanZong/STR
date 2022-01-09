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
    origin = "/home/zongzan/dist_dnn_training/STR/optimizer/logs/VGG16_250/Q-approx"
    q_appro = np.loadtxt(origin)
    new_r = np.zeros_like(q_appro, dtype=int)
    for i in range(q_appro.shape[0]):
        for j in range(q_appro.shape[1]):
            if q_appro[i, j] == 1:
                new_r[j + 1, j] = 1
    
    np.savetxt("/home/zongzan/dist_dnn_training/STR/optimizer/logs/VGG16_250/P-approx", new_r, fmt="%i")



def find_overlap_selected():
    dynprog = np.loadtxt("/home/zongzan/dist_dnn_training/STR/optimizer/logs/MobileNet_400/Q-dynprog")
    capuchin = np.loadtxt("/home/zongzan/dist_dnn_training/STR/optimizer/logs/MobileNet_400/Q-capuchin")
    str = np.loadtxt("/home/zongzan/dist_dnn_training/STR/optimizer/logs/MobileNet_400/PrunedQ")
    dynprog_col = np.sum(dynprog, axis=0)
    str_col = np.sum(str, axis=0)
    capuchin_col = np.sum(capuchin, axis=0)

    str_selected_ind = [i for i, sum in enumerate(str_col) if sum > 0]
    dynprog_selected_ind = [i for i, sum in enumerate(dynprog_col) if sum > 0]
    capuchin_selected_ind = [i for i, sum in enumerate(capuchin_col) if sum > 0]

    str_dynprog = len(set(str_selected_ind).intersection(dynprog_selected_ind)) / len(dynprog_selected_ind)
    str_capuchin = len(set(str_selected_ind).intersection(capuchin_selected_ind)) / len(capuchin_selected_ind)
    print("STR covers DYNPROG's selection by {}".format(str_dynprog))
    print("STR covers Capuchin's selection by {}".format(str_capuchin))


# gen_appro_checkmate_r()
fix_str_p()
# find_overlap_selected()