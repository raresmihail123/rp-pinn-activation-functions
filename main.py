"""
********************************************************************************
main file to execute your program
********************************************************************************
"""

import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from pinn import PINN
from config_gpu import config_gpu
from prp_dat import func_u0, func_ub, prp_grd, prp_dataset
from params import params
from make_fig import plot_sol0, plot_sol1
from plot_hist import *

def main():
    config_gpu(gpu_flg = 1)

    tmin, tmax =  0., 1.
    xmin, xmax = -1., 1.

    in_dim, out_dim, width, depth, \
        w_init, b_init, act, \
        lr, opt, \
        f_scl, laaf, \
        rho, nu, \
        w_dat, w_pde, \
        f_mntr, r_seed, \
        n_epch, n_btch, c_tol, \
        N_0, N_b, N_r, af_method = params()

    t_0, x_0, t_b, x_b, t_r, x_r = prp_dataset(tmin, tmax, xmin, xmax, N_0, N_b, N_r)
    u_0 = func_u0(x_0)
    u_b = func_ub(x_b)
    thresholds = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 0]
    pinn = PINN(t_0, x_0, u_0,
                t_b, x_b, u_b,
                t_r, x_r,
                Rm = in_dim, Rn = out_dim, Rl = width, depth = depth, af_method=af_method, activ = "sigmoid", BN = False,
                w_init = "glorot_normal", b_init = "zeros",
                lr = lr, opt = opt, w_0 = 1., w_b = 1., w_r = 1.,
                f_mntr = 10, r_seed = 1234)

    with tf.device("/device:GPU:0"):
        start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        pinn.train(start_time, thresholds, c_tol, n_epch, n_btch)

    plt.figure(figsize=(8,4))
    plt.plot(pinn.ep_log, pinn.loss_log, alpha=.7)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.ylim(1e-5, 1e2)  # Set the y-axis scale explicitly
    file_path_loss_curve = ""
    file_path_pde = ""
    file_path_pde_loss = ""
    if af_method == 0:
        file_path_loss_curve = f"figures/no_af/{start_time}_{pinn.activ}/loss_curve.png"
        file_path_pde = f"figures/no_af/{start_time}_{pinn.activ}/pde.png"
        file_path_pde_loss = f"figures/no_af/{start_time}_{pinn.activ}/pde_loss.png"
        file_path_parameters = f"figures/no_af/parameters.txt"
        plt.title(f"Loss history 1D Burgers {pinn.activ}")
    elif af_method == 1:
        file_path_loss_curve = f"figures/laaf/{start_time}/figure.png"
        file_path_pde = f"figures/laaf/{start_time}/pde.png"
        file_path_pde_loss = f"figures/laaf/{start_time}/pde_loss.png"
        file_path_parameters = f"figures/laaf/parameters.txt"
        plt.title("Loss history 1D Burgers LAAF")
    elif af_method == 2:
        file_path_loss_curve = f"figures/n_laaf/{start_time}/figure.png"
        file_path_pde = f"figures/n_laaf/{start_time}/pde.png"
        file_path_pde_loss = f"figures/n_laaf/{start_time}/pde_loss.png"
        file_path_parameters = f"figures/n_laaf/parameters.txt"
        plt.title("Loss history 1D Burgers N-LAAF")
    plt.grid(alpha=.5)
    plt.savefig(file_path_loss_curve, dpi=300, bbox_inches='tight')
    plt.show()
    # PINN inference
    nt = int(1e2) + 1
    nx = int(1e2) + 1
    t, x, TX = prp_grd(
        tmin, tmax, nt,
        xmin, xmax, nx
    )
    t0 = time.time()
    u_hat, gv_hat = pinn.predict(t, x)
    t1 = time.time()
    elps = t1 - t0
    print("elapsed time for PINN inference:", elps, "(sec)", elps / 60., "(min)")
    plot_sol1(TX, u_hat .numpy(), -1, 1, .25, file_path_pde)
    plot_sol1(TX, gv_hat.numpy(), -1, 1, .25, file_path_pde_loss)

if __name__ == "__main__":
    main()
