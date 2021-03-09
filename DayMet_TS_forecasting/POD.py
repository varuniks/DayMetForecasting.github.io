from Config import *
import numpy as np
from numpy import linalg as LA
import statsmodels
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Load DayMet dataset
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def load_snapshots_pod():

    # Two years of training
    tmax_train = np.load('./Data/daymet_v3_tmax_2014_na_tmax.npy',allow_pickle=True)
    tmax_train = np.concatenate((tmax_train,np.load('./Data/daymet_v3_tmax_2015_na_tmax.npy',allow_pickle=True)),axis=0)
    
    # One year each of validation and testing
    tmax_valid = np.load('./Data/daymet_v3_tmax_2016_na_tmax.npy',allow_pickle=True)
    tmax_test = np.load('./Data/daymet_v3_tmax_2017_na_tmax.npy',allow_pickle=True)

    for i in range(np.shape(tmax_train)[0]):
        tmax_train[i] = gaussian_filter(tmax_train[i],sigma=1)

    for i in range(np.shape(tmax_valid)[0]):
        tmax_valid[i] = gaussian_filter(tmax_valid[i],sigma=1)
        tmax_test[i] = gaussian_filter(tmax_test[i],sigma=1)

    dim_0 = np.shape(tmax_train)[0]
    dim_0_v = np.shape(tmax_valid)[0]
    dim_0_t = np.shape(tmax_test)[0]


    dim_1 = np.shape(tmax_train)[1]
    dim_2 = np.shape(tmax_train)[2]

    # Get rid of oceanic points with mask
    mask = np.zeros(shape=(dim_1,dim_2),dtype='bool')

    for i in range(dim_1):
        for j in range(dim_2):
            if tmax_train[0,i,j] > -1000:
                mask[i,j] = 1

    mask = mask.flatten()
    tmax_train = tmax_train.reshape(dim_0,dim_1*dim_2)
    tmax_valid = tmax_valid.reshape(dim_0_v,dim_1*dim_2)
    tmax_test = tmax_test.reshape(dim_0_t,dim_1*dim_2)

    tmax_train = tmax_train[:,mask]
    tmax_valid = tmax_valid[:,mask]
    tmax_test = tmax_test[:,mask]

    np.save('./Data/mask',mask)
    tmax_mean = np.mean(tmax_train,axis=0)

    tmax_train = (tmax_train-tmax_mean)
    tmax_valid = (tmax_valid-tmax_mean)
    tmax_test = (tmax_test-tmax_mean)

    np.save('./Coefficients/Snapshot_Mean.npy',tmax_mean)
    
    return np.transpose(tmax_train), np.transpose(tmax_valid), np.transpose(tmax_test)


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Load prepared coefficients
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def load_coefficients_pod():
    phi = np.load('./Coefficients/POD_Modes.npy')
    cf = np.load('./Coefficients/Coeffs_train.npy')
    cf_v = np.load('./Coefficients/Coeffs_valid.npy')
    cf_t = np.load('./Coefficients/Coeffs_test.npy')
    smean = np.load('./Coefficients/Snapshot_Mean.npy')

    # Do truncation
    phi = phi[:,0:num_modes] # Columns are modes
    cf = cf[0:num_modes,:] #Columns are time, rows are modal coefficients
    cf_v = cf_v[0:num_modes,:] #Columns are time, rows are modal coefficients
    cf_t = cf_t[0:num_modes,:] #Columns are time, rows are modal coefficients

    # Lowess filtering
    if perform_lowess:
        arr_len = np.shape(cf)[0]
        for i in range(np.shape(cf)[1]):
            cf[:,i] = lowess(cf[:,i], np.arange(arr_len), frac=0.3, return_sorted=False)

        arr_len = np.shape(cf_v)[0]
        for i in range(np.shape(cf_v)[1]):
            cf_v[:,i] = lowess(cf_v[:,i], np.arange(arr_len), frac=0.3, return_sorted=False)
            cf_t[:,i] = lowess(cf_t[:,i], np.arange(arr_len), frac=0.3, return_sorted=False)


    return phi, cf, cf_v, cf_t, smean


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Generate POD basis
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def generate_pod_bases(snapshot_matrix_train,snapshot_matrix_valid,snapshot_matrix_test,num_modes): #Mean removed
    '''
    Takes input of a snapshot matrix and computes POD bases
    Outputs truncated POD bases and coefficients
    '''
    new_mat = np.matmul(np.transpose(snapshot_matrix_train),snapshot_matrix_train)

    w,v = LA.eig(new_mat)

    # Bases
    phi = np.real(np.matmul(snapshot_matrix_train,v))
    trange = np.arange(np.shape(snapshot_matrix_train)[1])
    phi[:,trange] = phi[:,trange]/np.sqrt(w[:])

    coefficient_matrix = np.matmul(np.transpose(phi),snapshot_matrix_train)
    coefficient_matrix_valid = np.matmul(np.transpose(phi),snapshot_matrix_valid)
    coefficient_matrix_test = np.matmul(np.transpose(phi),snapshot_matrix_test)

    # Output amount of energy retained
    print('Amount of energy retained:',np.sum(w[:num_modes])/np.sum(w))
    input('Press any key to continue')

    np.save('./Coefficients/POD_Modes.npy',phi)
    np.save('./Coefficients/Coeffs_train.npy',coefficient_matrix)
    np.save('./Coefficients/Coeffs_valid.npy',coefficient_matrix_valid)
    np.save('./Coefficients/Coeffs_test.npy',coefficient_matrix_test)

    return phi, coefficient_matrix, coefficient_matrix_valid, coefficient_matrix_test


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Plot POD modes
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def plot_pod_modes(phi,mode_num):
    plt.figure()
    plt.plot(phi[:,mode_num])
    plt.show()

def calculate_max_lyapunov_exp(t_data, p_data, wlen, plot, mode=0):
    n_modes = p_data.shape[0]
    n_points = p_data.shape[1]
    rval = np.arange(0, n_points, wlen)
    p_lbd = [] 
    t_lbd = [] 
    for m in range(n_modes):
        p_lbd.append([]) 
        t_lbd.append([]) 
        for t in range(0,n_points,wlen):
            p_diff = np.log(np.abs(np.diff(p_data[m,t:t+wlen])))
            t_diff = np.log(np.abs(np.diff(t_data[m,t:t+wlen])))
            p_lbd[m].append(np.mean(p_diff))
            t_lbd[m].append(np.mean(t_diff))

    p_max_le = np.max(p_lbd,axis=0)
    t_max_le = np.max(t_lbd,axis=0)
    
    if plot:
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,7))
        xticks = np.linspace(0, n_points-1, n_points)
        # zero line
        zero = [0]*n_points
        ax.plot(xticks, zero, 'b-')
        # plot lyapunov
        ax.plot(rval, p_max_le, 'r-', linewidth = 1, label = 'Lyapunov exponent Pred')
        ax.plot(rval, t_max_le, 'g-', linewidth = 1, label = 'Lyapunov exponent Truth')
        ax.grid('on')
        ax.set_xlabel('test samples')
        ax.set_ylabel('lyapunov exponent')
        ax.legend(loc='best')
        ax.set_title('Max Lyapunov exponent for Truth Vs Predict')
        plt.show()

def calculate_lyapunov_exp(t_data, p_data, wlen, plot, mode=0):
    n_modes = p_data.shape[0]
    n_points = p_data.shape[1]
    rval = np.arange(0, n_points, wlen)
    for m in range(n_modes):
        p_lbd = [] 
        t_lbd = [] 
        for t in range(0,n_points,wlen):
            p_diff = np.log(np.abs(np.diff(p_data[m,t:t+wlen])))
            t_diff = np.log(np.abs(np.diff(t_data[m,t:t+wlen])))
            p_lbd.append(np.mean(p_diff))
            t_lbd.append(np.mean(t_diff))
        if plot:
            fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(10,7))
            #fig = plt.figure(figsize=(10,7))
            #ax1 = fig.add_subplot(1,1,1)
            #ax1 = fig.add_subplot(1,1,1)
            xticks = np.linspace(0, n_points-1, n_points)
            # zero line
            zero = [0]*n_points
            ax[0].plot(xticks, zero, 'b-')
            ax[1].plot(xticks, zero, 'b-')
            # plot pred and true  map
            ax[0].plot(xticks, p_data[m,:], 'r.',alpha = 0.3, label = 'Pred')
            ax[0].plot(xticks, t_data[m,:], 'g.',alpha = 0.3, label = 'Truth')
            ax[0].set_xlabel('test samples')
            ax[0].set_ylabel('Coeff values')
            # plot lyapunov
            ax[1].plot(rval, p_lbd, 'r-', linewidth = 1, label = 'Lyapunov exponent Pred')
            ax[1].plot(rval, t_lbd, 'g-', linewidth = 1, label = 'Lyapunov exponent Truth')
            ax[1].grid('on')
            ax[1].set_xlabel('test samples')
            ax[1].set_ylabel('lyapunov exponent')
            ax[1].legend(loc='best')
            ax[1].legend(loc='best')
            if mode == 0:
                ax[0].set_title('mode '+ str(m) +' Truth Vs Predict for time series ')
                ax[1].set_title('mode '+ str(m) +' Lyapunov exponent for Truth Vs Predict')
            else:
                ax[0].set_title('mode '+ str(mode) +' Truth Vs Predict for time series ')
                ax[1].set_title('mode '+ str(mode) +' Lyapunov exponent for Truth Vs Predict')
            plt.show()


def plot_lyapunov(true_s, predict_s):
    if len(true_s) == len(predict_s):
        n_points = len(true_s)
    else: 
        print(f"length of true and predicted series do not match")
        return
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,7))
    xticks = np.linspace(0, n_points-1, n_points)
    # zero line
    zero = [0]*n_points
    ax.plot(xticks, zero, 'b-')
    # plot lyapunov
    ax.plot(xticks, predict_s, 'r-', linewidth = 1, label = 'Lyapunov exponent Pred')
    ax.plot(xticks, true_s, 'g-', linewidth = 1, label = 'Lyapunov exponent Truth')
    ax.grid('on')
    ax.set_xlabel('After k steps')
    ax.set_ylabel('<n[d(k)]>')
    ax.legend(loc='best')
    ax.set_title('lyapunov exponent of Truth Vs Predict for primary mode ')
    plt.show()


def d(series,i,j):
    return abs(series[i]-series[j])

def Lyapunov_calc(series): 
    N=len(series)
    #eps= abs(np.max(series) - np.min(series))
    eps= 0.001
    dlist=[[] for i in range(N)]
    n=0 #number of nearby pairs found
    for i in range(N):
        for j in range(i+1,N):
            if d(series,i,j) < eps:
                n+=1
            for k in range(min(N-i,N-j)):
                dlist[k].append(np.log(d(series,i+k,j+k)))
    lyapunov = []
    for i in range(len(dlist)):
        if len(dlist[i]):
            lyapunov.append(sum(dlist[i])/len(dlist[i]))
    print(f"lyapunov shape : {len(lyapunov)}")
    return lyapunov


if __name__ == '__main__':
    pass
