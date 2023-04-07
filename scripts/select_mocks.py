import numpy as np
import os
import csv
import matplotlib.pyplot as plt

def process_fun(name, true_eq, nx, frac_sigx, all_compl, imax, all_new_z):

    # (1) Load the data
    fname = f'{name}_{nx}_{frac_sigx}'
    dirname = f'output/output_{fname}/' 
    print(dirname)
    
    res = []
    fun = []
    params = []
    store_comp = []
    for i, compl in enumerate(all_compl):
        fname = dirname + 'final_%i.dat'%compl
        if os.path.isfile(fname):
            with open(fname, "r") as f:
                reader = csv.reader(f, delimiter=';')
                data = [row for row in reader]

                res += [d[2:7] for d in data]
                fun += [d[1] for d in data]
                params += [d[7:] for d in data]
                store_comp += [compl] * len(data)
    res = np.array(res, dtype=float)
    params = np.array(params, dtype=float)
    store_comp = np.array(store_comp, dtype=int)

    # (2) Remove low-ranked functions and sort data
    m = np.argsort(res[:,0], kind='stable')
    res = res[m,:]
    params = params[m,:]
    store_comp = store_comp[m]
    fun = [fun[i] for i in m]
    res = res[:imax,:]
    params = params[:imax,:]
    store_comp = store_comp[:imax]
    fun = fun[:imax]

    # (3) Remove duplicates by likelihood
    _, uniq_idx = np.unique(res[:,2], return_index=True)
    uniq_idx = np.sort(uniq_idx)
    fun = [fun[i] for i in uniq_idx]
    res = res[uniq_idx,:]
    store_comp = store_comp[uniq_idx]
    params = params[uniq_idx,:]

    # (4) Remove duplicates by name
    new_fun = []
    uniq_idx = []
    for i in range(len(fun)):
        if fun[i] not in new_fun:
            uniq_idx.append(i)
            new_fun.append(fun[i])
    print(len(fun), len(new_fun))
    fun = new_fun
    res = res[uniq_idx,:]
    params = params[uniq_idx,:]
    store_comp = store_comp[uniq_idx]

    # (5) Separate into separate parts
    m = params!=0
    p = np.sum(m, axis=1)
    p_term = -p/2 * np.log(3)  # -p/2 log(3)
    logtheta = params.copy()
    logtheta[~m] = 1  # so do log(1)=0 when no parameter
    logtheta = np.log(np.abs(logtheta))
    logtheta = np.sum(logtheta, axis=1)  # sum_i log(theta_i)
    fish_term = res[:,3] - logtheta - p_term
    logL = res[:,2]
    functional = res[:,4]

    def get_delta_DL(new_nz, method=1):
        
        if method == 1:
            # Max likelihood
            DL = (new_nz / nx) * logL
            
            m = np.argsort(DL, kind='stable')
            f0, f1 = fun[m[0]], fun[m[1]]
            DL0, DL1 = DL[m[:2]]
            
        elif method == 2:
            # Max dlogL / dc
            
            DL = np.empty(len(all_compl))
            new_fun = [None] * len(all_compl)
            for i,c in enumerate(all_compl):
                m = store_comp==c
                idx = np.arange(len(logL))[m]
                ll = (new_nz / nx) * logL[m]
                DL[i] = np.amin(ll)
                new_fun[i] = fun[idx[np.argmin(ll)]]
            DL = DL[1:] -- DL[:-1]
            new_fun = new_fun[1:]
            
            m = np.argsort(DL, kind='stable')
            print(m[0], m[1])
            f0, f1 = new_fun[m[0]], new_fun[m[1]]
            DL0, DL1 = DL[m[:2]]
            
        elif method == 3:
            # Prescription of Bartlett et al. 2022
            DL = (new_nz / nx) * logL + p/2 * np.log(new_nz / nx) + res[:,4] + res[:,3]
            
            m = np.argsort(DL, kind='stable')
            f0, f1 = fun[m[0]], fun[m[1]]
            DL0, DL1 = DL[m[:2]]
            
        elif method == 4:
            pass
        elif method == 5:
            pass
        elif method == 6:
            pass

        if f0 == true_eq:
            toptwo = True
            DL = DL1 - DL0
        elif f1 == true_eq:
            toptwo = True
            DL = DL0 - DL1
        else:
            toptwo = False
            i = list(m).index(0)
            DL = DL0 - DL[0]
        return toptwo, DL

    all_DL = np.empty(len(all_new_z))
    all_toptwo = np.zeros(len(all_new_z), dtype=bool)
    for i, new_nz in enumerate(all_new_z[-2:]):
        all_toptwo[i], all_DL[i] = get_delta_DL(new_nz, method=2)
        print('%.1f'%new_nz, all_toptwo[i], '%.2e'%all_DL[i])
        quit()
       
    fig, ax = plt.subplots()
    ax.semilogx(all_new_z, all_DL, marker='.')
    ax.axhline(y=0, color='k')
    ax.set_xlabel('Number of data points')
    ax.set_ylabel('Preference for truth')
    ax.set_title(name + ': ' + true_eq + ', frac_sigx = ' + str(frac_sigx))
    plt.show()

    return


nx = 10000
frac_sigx = 0.5
#name = 'nguyen_8'; true_eq = 'sqrt(x)'
name = 'korns_4'; true_eq = 'a0 + a1*sin(x)'
all_new_z = np.logspace(0, 3, 10)
all_compl = np.arange(1, 7)
#imax = 1000
imax = -1

process_fun(name, true_eq, nx, frac_sigx, all_compl, imax, all_new_z)
