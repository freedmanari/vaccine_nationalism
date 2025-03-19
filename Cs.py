import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import numpy as np
from multiprocessing import Pool

### default parameter values
gamma_def = 1/10
lambda_def = 2/365
R0_def = 2
beta_def = R0_def * gamma_def

f_B_def = 1
sigma_def = 1.1
mu_def = 2
nu_def = 1/365
eta_def = .05
delta_def = .05/365
rho_def = .05

pop_size_def = 1e6
epsilon_def = 1e-6

### function encoding ODEs between timesteps
def model(y, t,
          f_A, f_B,
          beta, gamma, lambd,
          sigma, mu, phi,
          nu, eta,
          kA, kB, pw_init, A_weights, B_weights):
    Sw_1A, Sw_2A, Iw_wA, Rw_A, \
    Sw_1B, Sw_2B, Iw_wB, Rw_B, \
    pw, pA, pB, C_A, C_B = y[:13]
    
    SA_1As = y[13:(13 + kA)]
    SA_2As = y[(13 + kA):(13 + 2*kA)]
    IA_wAs = y[(13 + 2*kA):(13 + 3*kA)]
    IA_vAs = y[(13 + 3*kA):(13 + 4*kA)]
    RA_As = y[(13 + 4*kA):(13 + 5*kA)]
    SA_1Bs = y[(13 + 5*kA):(13 + 6*kA)]
    SA_2Bs = y[(13 + 6*kA):(13 + 7*kA)]
    IA_wBs = y[(13 + 7*kA):(13 + 8*kA)]
    IA_vBs = y[(13 + 8*kA):(13 + 9*kA)]
    RA_Bs = y[(13 + 9*kA):(13 + 10*kA)]
    
    SB_1As = y[(13 + 10*kA):(13 + 10*kA + kB)]
    SB_2As = y[(13 + 10*kA + kB):(13 + 10*kA + 2*kB)]
    IB_wAs = y[(13 + 10*kA + 2*kB):(13 + 10*kA + 3*kB)]
    IB_vAs = y[(13 + 10*kA + 3*kB):(13 + 10*kA + 4*kB)]
    RB_As = y[(13 + 10*kA + 4*kB):(13 + 10*kA + 5*kB)]
    SB_1Bs = y[(13 + 10*kA + 5*kB):(13 + 10*kA + 6*kB)]
    SB_2Bs = y[(13 + 10*kA + 6*kB):(13 + 10*kA + 7*kB)]
    IB_wBs = y[(13 + 10*kA + 7*kB):(13 + 10*kA + 8*kB)]
    IB_vBs = y[(13 + 10*kA + 8*kB):(13 + 10*kA + 9*kB)]
    RB_Bs = y[(13 + 10*kA + 9*kB):(13 + 10*kA + 10*kB)]
    
    beta_w = beta
    beta_v = sigma * beta

    
    dSw_1A = -beta_w * ((1-eta) * Iw_wA + eta * Iw_wB) * Sw_1A - (1 - f_A) * nu * Sw_1A
    dSw_2A = lambd * Rw_A
    dIw_wA = beta_w * ((1-eta) * Iw_wA + eta * Iw_wB) * Sw_1A - gamma * Iw_wA
    dRw_A = gamma * Iw_wA - lambd * Rw_A + (1 - f_A) * nu * Sw_1A
     
    dSw_1B = -beta_w * ((1-eta) * Iw_wB + eta * Iw_wA) * Sw_1B - min(f_A, f_B) * nu * Sw_1B
    dSw_2B = lambd * Rw_B
    dIw_wB = beta_w * ((1-eta) * Iw_wB + eta * Iw_wA) * Sw_1B - gamma * Iw_wB
    dRw_B = gamma * Iw_wB - lambd * Rw_B + min(f_A, f_B) * nu * Sw_1B
    

    dSA_1As = -beta_w * ((1-eta) * IA_wAs + eta * IA_wBs) * SA_1As - beta_v * ((1-eta) * IA_vAs + eta * IA_vBs) * SA_1As - (1 - f_A) * nu * SA_1As
    dSA_2As = lambd * RA_As - phi * beta_v * ((1-eta) * IA_vAs + eta * IA_vBs) * SA_2As
    dIA_wAs = beta_w * ((1-eta) * IA_wAs + eta * IA_wBs) * SA_1As - gamma * IA_wAs
    dIA_vAs = beta_v * ((1-eta) * IA_vAs + eta * IA_vBs) * (SA_1As + phi * SA_2As) - gamma * IA_vAs
    dRA_As = gamma * (IA_wAs + IA_vAs) - lambd * RA_As + (1 - f_A) * nu * SA_1As
    
    dSA_1Bs = -beta_w * ((1-eta) * IA_wBs + eta * IA_wAs) * SA_1Bs - beta_v * ((1-eta) * IA_vBs + eta * IA_vAs) * SA_1Bs - min(f_A, f_B) * nu * SA_1Bs
    dSA_2Bs = lambd * RA_Bs - phi * beta_v * ((1-eta) * IA_vBs + eta * IA_vAs) * SA_2Bs
    dIA_wBs = beta_w * ((1-eta) * IA_wBs + eta * IA_wAs) * SA_1Bs - gamma * IA_wBs
    dIA_vBs = beta_v * ((1-eta) * IA_vBs + eta * IA_vAs) * (SA_1Bs + phi * SA_2Bs) - gamma * IA_vBs
    dRA_Bs = gamma * (IA_wBs + IA_vBs) - lambd * RA_Bs + min(f_A, f_B) * nu * SA_1Bs
    
    
    dSB_1As = -beta_w * ((1-eta) * IB_wAs + eta * IB_wBs) * SB_1As - beta_v * ((1-eta) * IB_vAs + eta * IB_vBs) * SB_1As - (1 - f_A) * nu * SB_1As
    dSB_2As = lambd * RB_As - phi * beta_v * ((1-eta) * IB_vAs + eta * IB_vBs) * SB_2As
    dIB_wAs = beta_w * ((1-eta) * IB_wAs + eta * IB_wBs) * SB_1As - gamma * IB_wAs
    dIB_vAs = beta_v * ((1-eta) * IB_vAs + eta * IB_vBs) * (SB_1As + phi * SB_2As) - gamma * IB_vAs
    dRB_As = gamma * (IB_wAs + IB_vAs) - lambd * RB_As + (1 - f_A) * nu * SB_1As
    
    dSB_1Bs = -beta_w * ((1-eta) * IB_wBs + eta * IB_wAs) * SB_1Bs - beta_v * ((1-eta) * IB_vBs + eta * IB_vAs) * SB_1Bs - min(f_A, f_B) * nu * SB_1Bs
    dSB_2Bs = lambd * RB_Bs - phi * beta_v * ((1-eta) * IB_vBs + eta * IB_vAs) * SB_2Bs
    dIB_wBs = beta_w * ((1-eta) * IB_wBs + eta * IB_wAs) * SB_1Bs - gamma * IB_wBs
    dIB_vBs = beta_v * ((1-eta) * IB_vBs + eta * IB_vAs) * (SB_1Bs + phi * SB_2Bs) - gamma * IB_vBs
    dRB_Bs = gamma * (IB_wBs + IB_vBs) - lambd * RB_Bs + min(f_A, f_B) * nu * SB_1Bs

    
    rA = mu * beta_w * Sw_1A * ((1-eta) * Iw_wA + eta * Iw_wB) * np.maximum(0, 1 - gamma/(beta_v * (Sw_1A + phi * Sw_2A)))
    rB = mu * beta_w * Sw_1B * ((1-eta) * Iw_wB + eta * Iw_wA) * np.maximum(0, 1 - gamma/(beta_v * (Sw_1B + phi * Sw_2B)))
    
    dpw = -(rA + rB) * pw
    dpA = rA * pw
    dpB = rB * pw
    
    dJ_A = pw_init * Iw_wA + sum(A_weights * (IA_wAs + IA_vAs)) + sum(B_weights * (IB_wAs + IB_vAs))
    dJ_B = pw_init * Iw_wB + sum(A_weights * (IA_wBs + IA_vBs)) + sum(B_weights * (IB_wBs + IB_vBs))
    
    return np.concatenate(([dSw_1A, dSw_2A, dIw_wA, dRw_A,
                            dSw_1B, dSw_2B, dIw_wB, dRw_B,
                            dpw, dpA, dpB, dJ_A, dJ_B],
                           dSA_1As, dSA_2As, dIA_wAs, dIA_vAs, dRA_As,
                           dSA_1Bs, dSA_2Bs, dIA_wBs, dIA_vBs, dRA_Bs,
                           dSB_1As, dSB_2As, dIB_wAs, dIB_vAs, dRB_As,
                           dSB_1Bs, dSB_2Bs, dIB_wBs, dIB_vBs, dRB_Bs))


### solves the model ODEs between each timestep, gets raw infection costs over time J_A and J_B.
### if the wild-type adaptive potential in a country is large enough in last timestep, a new
### parallel scenario with variant emerging in the next timestep is added to and tracked by the ODEs
def get_Js(f_A, f_B=f_B_def,
           beta=beta_def, gamma=gamma_def, lambd=lambda_def,
           sigma=sigma_def, mu=mu_def, phi=None,
           nu=nu_def, eta=eta_def,
           pop_size=pop_size_def, epsilon=epsilon_def,
           tstep=7, tsteps=1000, ts_per_tstep=1, track_vars=False):
    tstep_ts = np.linspace(0, tsteps * tstep, tsteps + 1)
    all_ts = np.linspace(0, tsteps * tstep, tsteps * ts_per_tstep + 1)
    init = 1 / pop_size
    if phi is None:
        phi = 1 / (sigma * beta / gamma) ### sets R0 for the variant among recovered individuals to 1, to standardize across different wild-type R0s
    
    kA = 0
    kB = 0
    pw_init = 1
    A_weights = []
    B_weights = []
    
    y0 = [1 - init, 0, init, 0,
          1 - init, 0, init, 0,
          1, 0, 0, 0, 0]

    if track_vars:
        Sw_1A = [1 - init]
        Sw_2A = [0]
        Iw_wA = [init]
        Rw_A = [0]
        Sw_1B = [1 - init]
        Sw_2B = [0]
        Iw_wB = [init]
        Rw_B = [0]
        pw = [pw_init]
        pA = [0]
        pB = [0]

        SA_1As = [[]]
        SA_2As = [[]]
        IA_wAs = [[]]
        IA_vAs = [[]]
        RA_As = [[]]
        SA_1Bs = [[]]
        SA_2Bs = [[]]
        IA_wBs = [[]]
        IA_vBs = [[]]
        RA_Bs = [[]]
        SB_1As = [[]]
        SB_2As = [[]]
        IB_wAs = [[]]
        IB_vAs = [[]]
        RB_As = [[]]
        SB_1Bs = [[]]
        SB_2Bs = [[]]
        IB_wBs = [[]]
        IB_vBs = [[]]
        RB_Bs = [[]]
        
    J_A = [0]
    J_B = [0]

    for i in range(1, tsteps + 1):
        ts = np.linspace((i-1) * tstep, i * tstep, ts_per_tstep + 1)
        params = (f_A, f_B,
                  beta, gamma, lambd,
                  sigma, mu, phi,
                  nu, eta,
                  kA, kB, pw_init, A_weights, B_weights)
        
        odes = scipy.integrate.odeint(model, y0, ts, args = params)

        if track_vars:
            Sw_1A = np.append(Sw_1A, odes[1:,0])
            Sw_2A = np.append(Sw_2A, odes[1:,1])
            Iw_wA = np.append(Iw_wA, odes[1:,2])
            Rw_A = np.append(Rw_A, odes[1:,3])
            Sw_1B = np.append(Sw_1B, odes[1:,4])
            Sw_2B = np.append(Sw_2B, odes[1:,5])
            Iw_wB = np.append(Iw_wB, odes[1:,6])
            Rw_B = np.append(Rw_B, odes[1:,7])
            pw = np.append(pw, odes[1:,8])
            pA = np.append(pA, odes[1:,9])
            pB = np.append(pB, odes[1:,10])

            SA_1As = np.append(SA_1As, odes[1:,13:(13 + kA)], 0)
            SA_2As = np.append(SA_2As, odes[1:,(13 + kA):(13 + 2*kA)], 0)
            IA_wAs = np.append(IA_wAs, odes[1:,(13 + 2*kA):(13 + 3*kA)], 0)
            IA_vAs = np.append(IA_vAs, odes[1:,(13 + 3*kA):(13 + 4*kA)], 0)
            RA_As = np.append(RA_As, odes[1:,(13 + 4*kA):(13 + 5*kA)], 0)
            SA_1Bs = np.append(SA_1Bs, odes[1:,(13 + 5*kA):(13 + 6*kA)], 0)
            SA_2Bs = np.append(SA_2Bs, odes[1:,(13 + 6*kA):(13 + 7*kA)], 0)
            IA_wBs = np.append(IA_wBs, odes[1:,(13 + 7*kA):(13 + 8*kA)], 0)
            IA_vBs = np.append(IA_vBs, odes[1:,(13 + 8*kA):(13 + 9*kA)], 0)
            RA_Bs = np.append(RA_Bs, odes[1:,(13 + 9*kA):(13 + 10*kA)], 0)

            SB_1As = np.append(SB_1As, odes[1:,(13 + 10*kA):(13 + 10*kA + kB)], 0)
            SB_2As = np.append(SB_2As, odes[1:,(13 + 10*kA + kB):(13 + 10*kA + 2*kB)], 0)
            IB_wAs = np.append(IB_wAs, odes[1:,(13 + 10*kA + 2*kB):(13 + 10*kA + 3*kB)], 0)
            IB_vAs = np.append(IB_vAs, odes[1:,(13 + 10*kA + 3*kB):(13 + 10*kA + 4*kB)], 0)
            RB_As = np.append(RB_As, odes[1:,(13 + 10*kA + 4*kB):(13 + 10*kA + 5*kB)], 0)
            SB_1Bs = np.append(SB_1Bs, odes[1:,(13 + 10*kA + 5*kB):(13 + 10*kA + 6*kB)], 0)
            SB_2Bs = np.append(SB_2Bs, odes[1:,(13 + 10*kA + 6*kB):(13 + 10*kA + 7*kB)], 0)
            IB_wBs = np.append(IB_wBs, odes[1:,(13 + 10*kA + 7*kB):(13 + 10*kA + 8*kB)], 0)
            IB_vBs = np.append(IB_vBs, odes[1:,(13 + 10*kA + 8*kB):(13 + 10*kA + 9*kB)], 0)
            RB_Bs = np.append(RB_Bs, odes[1:,(13 + 10*kA + 9*kB):(13 + 10*kA + 10*kB)], 0)

        J_A = np.append(J_A, odes[-1,11] - odes[0,11])
        J_B = np.append(J_B, odes[-1,12] - odes[0,12])

        if i == tsteps:
            break

        y0 = list(odes[-1])
        pw_init = y0[8]

        ### there was enough evolutionary potential in country A last timestep for variant to possibly emerge in this timestep
        ### keeps track of new set of variables in scenario where variant emerges this timestep in country A
        if odes[-1,9] - odes[0,9] > epsilon:
            if track_vars:
                SA_1As = np.append(SA_1As, np.transpose([Sw_1A]), 1)
                SA_2As = np.append(SA_2As, np.transpose([Sw_2A]), 1)
                IA_wAs = np.append(IA_wAs, np.transpose([Iw_wA]), 1)
                IA_vAs = np.append(IA_vAs, [[0]] * i * ts_per_tstep + [[init]], 1)
                RA_As = np.append(RA_As, np.transpose([Rw_A]), 1)

                SA_1Bs = np.append(SA_1Bs, np.transpose([Sw_1B]), 1)
                SA_2Bs = np.append(SA_2Bs, np.transpose([Sw_2B]), 1)
                IA_wBs = np.append(IA_wBs, np.transpose([Iw_wB]), 1)
                IA_vBs = np.append(IA_vBs, [[0]] * (i * ts_per_tstep + 1), 1)
                RA_Bs = np.append(RA_Bs, np.transpose([Rw_B]), 1)

                SA_1As[-1,-1] -= Sw_1A[-1] / (Sw_1A[-1] + Sw_2A[-1]) * init
                SA_2As[-1,-1] -= Sw_2A[-1] / (Sw_1A[-1] + Sw_2A[-1]) * init
                if SA_1As[-1,-1] < 0 or SA_2As[-1,-1] < 0:
                    raise Exception("Not enough susceptibles in country A for variant to emerge")

                y0[13:(13 + 10*kA)] = \
                    np.concatenate((SA_1As[-1], SA_2As[-1], IA_wAs[-1], IA_vAs[-1], RA_As[-1],
                                    SA_1Bs[-1], SA_2Bs[-1], IA_wBs[-1], IA_vBs[-1], RA_Bs[-1]))
            else:
                y0[13:(13 + 10*kA)] = \
                    np.ndarray.flatten(np.append(np.reshape(y0[13:(13 + 10*kA)], (10,-1)),
                                                 [[odes[-1,0] - odes[-1,0] / (odes[-1,0] + odes[-1,1]) * init],
                                                  [odes[-1,1] - odes[-1,1] / (odes[-1,0] + odes[-1,1]) * init],
                                                  [odes[-1,2]], [init], [odes[-1,3]],
                                                  [odes[-1,4]], [odes[-1,5]], [odes[-1,6]], [0], [odes[-1,7]]],
                                                 1))

            kA += 1
            A_weights.append(odes[-1,9] - odes[0,9])

        ### there was enough evolutionary potential in country B last timestep for variant to possibly emerge in this timestep
        ### keeps track of new set of variables in scenario where variant emerges this timestep in country B
        if odes[-1,10] - odes[0,10] > epsilon:
            if track_vars:
                SB_1As = np.append(SB_1As, np.transpose([Sw_1A]), 1)
                SB_2As = np.append(SB_2As, np.transpose([Sw_2A]), 1)
                IB_wAs = np.append(IB_wAs, np.transpose([Iw_wA]), 1)
                IB_vAs = np.append(IB_vAs, [[0]] * (i * ts_per_tstep + 1), 1)
                RB_As = np.append(RB_As, np.transpose([Rw_A]), 1)

                SB_1Bs = np.append(SB_1Bs, np.transpose([Sw_1B]), 1)
                SB_2Bs = np.append(SB_2Bs, np.transpose([Sw_2B]), 1)
                IB_wBs = np.append(IB_wBs, np.transpose([Iw_wB]), 1)
                IB_vBs = np.append(IB_vBs, [[0]] * i * ts_per_tstep + [[init]], 1)
                RB_Bs = np.append(RB_Bs, np.transpose([Rw_B]), 1)
                
                SB_1Bs[-1,-1] -= Sw_1B[-1] / (Sw_1B[-1] + Sw_2B[-1]) * init
                SB_2Bs[-1,-1] -= Sw_2B[-1] / (Sw_1B[-1] + Sw_2B[-1]) * init
                if SB_1Bs[-1,-1] < 0 or SB_2Bs[-1,-1] < 0:
                    raise Exception("Not enough susceptibles in country B for variant to emerge")

                y0[(13 + 10*kA):(13 + 10*kA + 10*kB)] = \
                    np.concatenate((SB_1As[-1], SB_2As[-1], IB_wAs[-1], IB_vAs[-1], RB_As[-1],
                                    SB_1Bs[-1], SB_2Bs[-1], IB_wBs[-1], IB_vBs[-1], RB_Bs[-1]))
            else:
                y0[(13 + 10*kA):(13 + 10*kA + 10*kB)] = \
                    np.ndarray.flatten(np.append(np.reshape(y0[(13 + 10*kA):(13 + 10*kA + 10*kB)], (10,-1)),
                                                 [[odes[-1,0]], [odes[-1,1]], [odes[-1,2]], [0], [odes[-1,3]],
                                                  [odes[-1,4] - odes[-1,4] / (odes[-1,4] + odes[-1,5]) * init],
                                                  [odes[-1,5] - odes[-1,5] / (odes[-1,4] + odes[-1,5]) * init],
                                                  [odes[-1,6]], [init], [odes[-1,7]]],
                                                 1))

            kB += 1
            B_weights.append(odes[-1,10] - odes[0,10])
            
    if track_vars:
        return tstep_ts, all_ts, J_A, J_B, \
               pw, pA, pB, A_weights, B_weights, \
               Sw_1A, Sw_2A, Iw_wA, Rw_A, \
               Sw_1B, Sw_2B, Iw_wB, Rw_B, \
               SA_1As, SA_2As, IA_wAs, IA_vAs, RA_As, \
               SA_1Bs, SA_2Bs, IA_wBs, IA_vBs, RA_Bs, \
               SB_1As, SB_2As, IB_wAs, IB_vAs, RB_As, \
               SB_1Bs, SB_2Bs, IB_wBs, IB_vBs, RB_Bs
    else:
        return tstep_ts, J_A, J_B


### gets raw infection costs over time J_A and J_B, then accumulates them with discount rate delta and
### prosociality factor rho (or set of multiple deltas/rhos) to get true costs C_A and C_B.
### approx_inf_t says whether an approximation is made to calculate C_A and C_B into infinite time,
### assuming equations have gone to equilibrium by the end of the simulation and will continue to be discounted at rate delta
def get_Cs(f_A, f_B=f_B_def,
           delta=delta_def, rho=rho_def,
           beta=beta_def, gamma=gamma_def, lambd=lambda_def,
           sigma=sigma_def, mu=mu_def, phi=None,
           nu=nu_def, eta=eta_def,
           pop_size=pop_size_def, epsilon=epsilon_def,
           tstep=7, tsteps=2000, ts_per_tstep=1, approx_inf_t=True, approx_frac=.1):
    
    ts, J_A, J_B = get_Js(f_A, f_B,
                             beta, gamma, lambd,
                             sigma, mu, phi,
                             nu, eta,
                             pop_size, epsilon,
                             tstep, tsteps, ts_per_tstep, track_vars=False)
    
    def calc_C_A(delt, rh):
        C_A = sum(np.exp(-delt*ts) * ((1-rh) * J_A + rh * J_B))
        if approx_inf_t:
            C_A += np.exp(-delt*(ts[-1]+tstep)) / (1 - np.exp(-delt*tstep)) * \
                   np.mean((1-rh) * J_A[-int(approx_frac * len(ts)):] + rh * J_B[-int(approx_frac * len(ts)):])
        return C_A
    def calc_C_B(delt, rh):
        C_B = sum(np.exp(-delt*ts) * ((1-rh) * J_B + rh * J_A))
        if approx_inf_t:
            C_B += np.exp(-delt*(ts[-1]+tstep)) / (1 - np.exp(-delt*tstep)) * \
                   np.mean((1-rh) * J_B[-int(approx_frac * len(ts)):] + rh * J_A[-int(approx_frac * len(ts)):])
        return C_B
    
    if isinstance(delta, (float,int)):
        if isinstance(rho, (float,int)):
            return calc_C_A(delta, rho), calc_C_B(delta, rho)
        else:
            return [calc_C_A(delta, rh) for rh in rho], [calc_C_B(delta, rh) for rh in rho]
    else:
        if isinstance(rho, (float,int)):
            return [calc_C_A(delt, rho) for delt in delta], [calc_C_B(delt, rho) for delt in delta]
        else:
            return [[calc_C_A(delt, rh) for rh in rho] for delt in delta], [[calc_C_B(delt, rh) for rh in rho] for delt in delta]




def func(f_B):
    return [get_Cs(f_A, f_B=f_B, mu=10) for f_A in np.linspace(0,1,201)]



if __name__ == '__main__':
    pool=Pool(processes=30)
    Cs = pool.map(func, np.linspace(0,1,201))
    with open('/scratch/gpfs/arisf/Cs.txt', 'w') as f: ### change file path
        f.write(str(Cs))
