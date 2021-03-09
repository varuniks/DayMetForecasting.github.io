import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--modes",help="Number of POD modes to retain (always needed)",type=int)
parser.add_argument("--train",help="Train LSTM",action='store_true')
parser.add_argument('--pod', help="Run a POD on the data", action='store_true')
parser.add_argument('--filter', help="Lowess filter coefficients", action='store_true')
parser.add_argument('--viz', help="Visualize reconstruction", action='store_true')
parser.add_argument("--epochs",help="Number of epochs for training",type=int)
parser.add_argument("--win",help="Length of forecast window (always needed)",type=int)
parser.add_argument("--batch",help="Batch size of LSTM training",type=int)
parser.add_argument("--fb",help="feedback from o/p",action='store_true')
parser.add_argument("--model",help="ML model",choices=['lstm', 'blstm', '1dconv','tcn', 'stcn', 'ED', 'ED2'])
args = parser.parse_args()

# Import and tweak configuration
import Config

# Number of modes
Config.num_modes = args.modes
# Number of epochs to train
Config.num_epochs = args.epochs
# Window length of forecast
Config.window_len = args.win
# Deployment mode
Config.train_mode = args.train # test or train
# Field visualization
Config.field_viz = args.viz
# Perform POD?
Config.perform_pod = args.pod
# Filter the coefficients
Config.perform_lowess = args.filter
# Batch size of LSTM training
Config.batchsize = args.batch
# feedback of previous o/p to next prediction
Config.fb = args.fb
# ML model to be applied to the raw data
Config.model = args.model
# Import the configuration finally
from Config import *
import matplotlib.pyplot as plt

# Import libraries
import numpy as np
np.random.seed(5)
import tensorflow as tf
tf.random.set_seed(5)
repeat_m = True
from POD import generate_pod_bases, plot_pod_modes, load_snapshots_pod, load_coefficients_pod, calculate_lyapunov_exp, Lyapunov_calc, plot_lyapunov 
from ML_model import model_for_dynamics, model_for_error, evaluate_rom_deployment, evaluate_rom_error, evaluate_rom_deployment_predicted_history
from ML_GP import fit_error_model
from Analyses import visualize_predictions, visualize_error, analyze_predictions

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Scaling
preproc_input = Pipeline([('minmaxscaler', MinMaxScaler())])
preproc_input1 = Pipeline([('Stdscaler', StandardScaler()), ('minmaxscaler', MinMaxScaler())])
e_scale = False
plot_e = True 
print(f"model to be applied {model}" )
def plot_dis(cf, label):
    if plot_e == False:
        return
    #plt.rcParams.update({'font.size': 22})
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    for m in range(num_modes-1):
        # Set up the plot
        ax = plt.subplot(2, 2, m + 1)

        # Draw the plot
        ax.hist(cf[m,:],color = 'g', edgecolor = 'black', bins=25)
        # Title and labels
        ax.set_title('Mode %d' % m, fontsize=24)
        ax.set_xlabel('Scaled Coefficients', fontsize=24)
        ax.set_ylabel('Frequency', fontsize=24)

    #plt.suptitle(label)
    plt.tight_layout()
    plt.show()

def save_file(cf, label):
    filename = './Error/cf' + '_'+ label
    np.save(filename, cf)
 
def save_error(cf, model, fb, label):
    if fb == True:
        filename = './Error/error_fb_' + model +'_'+ label
    else:
        filename = './Error/error_' + model +'_'+ label
    np.save(filename, cf)

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# This is the main file for POD-LSTM assessment
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    if train_mode:

        # Perform POD
        if perform_pod:
            # Snapshot collection
            sm_train, sm_valid, sm_test = load_snapshots_pod() # Note that columns of a snapshot/state are time always and a state vector is a column vector
            # Eigensolve
            generate_pod_bases(sm_train,sm_valid,sm_test,num_modes)
            phi, cf, cf_v, cf_t, _ = load_coefficients_pod()
        else:
            phi, cf, cf_v, cf_t, _ = load_coefficients_pod()

        # Need to scale the rows individually to ensure common scale - need to add validation here
        cf = np.transpose(preproc_input.fit_transform(np.transpose(cf)))
        cf_v = np.transpose(preproc_input.transform(np.transpose(cf_v)))
        cf_t = np.transpose(preproc_input.transform(np.transpose(cf_t)))

        #plot_dis(np.concatenate((cf,cf_v,cf_t),axis=1), '')
        # LSTM network
        model_d = model_for_dynamics(cf,cf_v,num_epochs,window_len,train_mode,model)
        
        #### training
        tsteps = np.shape(cf)[1]
        # get the prediction for training data
        if fb is False:
            _, pred_d = evaluate_rom_deployment(model_d,cf,tsteps,num_modes,window_len,model)
        else:
            _, pred_d = evaluate_rom_deployment_predicted_history(model_d,cf,tsteps,num_modes,window_len,model)
             
        pred_d = np.mean(pred_d, axis=-1)
        # get the error 
        cf_e = cf - pred_d

        #### validation 
        tsteps = np.shape(cf_v)[1]
        # get the prediction for training data
        if fb is False: 
            _, pred_vd = evaluate_rom_deployment(model_d,cf_v,tsteps,num_modes,window_len,model)
        else:
            _, pred_vd = evaluate_rom_deployment_predicted_history(model_d,cf_v,tsteps,num_modes,window_len,model)

        pred_vd = np.mean(pred_vd, axis=-1)
        # get the error 
        cf_ve = cf_v - pred_vd

    else:
        # Load data for testing
        phi, cf, cf_v, cf_t, smean = load_coefficients_pod()
        #plot_dis(cf, 'Truth-before Scaling')
        cf = np.transpose(preproc_input.fit_transform(np.transpose(cf)))
        cf_t = np.transpose(preproc_input.transform(np.transpose(cf_t)))
        cf_v = np.transpose(preproc_input.transform(np.transpose(cf_v)))
        
        # model dynamics on training data and validation data
        model_d = model_for_dynamics(cf,cf_v,num_epochs,window_len,train_mode,model)
        
        if fb is False:
            # evaluate model dynamics on training data 
            tsteps = np.shape(cf)[1]
            _, pred_d = evaluate_rom_deployment(model_d,cf,tsteps,num_modes,window_len,model)
            # evaluate model dynamics on validation and test data
            tsteps = np.shape(cf_v)[1]
            _, pred_vd = evaluate_rom_deployment(model_d,cf_v,tsteps,num_modes,window_len,model)
            _, pred_td =  evaluate_rom_deployment(model_d,cf_t,tsteps,num_modes,window_len,model)
        else:
            # evaluate model dynamics on training data 
            tsteps = np.shape(cf)[1]
            _, pred_d = evaluate_rom_deployment_predicted_history(model_d,cf,tsteps,num_modes,window_len,model)
            # evaluate model dynamics on validation and test data
            tsteps = np.shape(cf_v)[1]
            _, pred_vd = evaluate_rom_deployment_predicted_history(model_d,cf_v,tsteps,num_modes,window_len,model)
            _, pred_td = evaluate_rom_deployment_predicted_history(model_d,cf_t,tsteps,num_modes,window_len,model)
        
        pred_d = np.mean(pred_d, axis=-1)
        pred_vd = np.mean(pred_vd, axis=-1)
        pred_td = np.mean(pred_td, axis=-1)
        
        # cal the error from the first round of prediction - (error = truth-predicted)
        cf_e = cf - pred_d
        cf_ve = cf_v - pred_vd
        cf_te = cf_t - pred_td
        #plot_dis(pred_d, 'prediction')
        #plot_dis(cf_e, 'error')
        save_error(cf_e, model,  fb, 'train_before_EC') 
        save_error(cf_ve, model, fb,  'valid_before_EC') 
        save_error(cf_te, model, fb, 'test_before_EC') 

        #scale the error
        if e_scale:
            cf_e = np.transpose(preproc_input1.fit_transform(np.transpose(cf_e)))
            cf_ve = np.transpose(preproc_input1.transform(np.transpose(cf_ve)))
            cf_te = np.transpose(preproc_input1.transform(np.transpose(cf_te)))
            plot_dis(cf_e, 'error after scaling')
        # get the model for error correction on predicted data  
        tr_data_for_EC = pred_d  # 
        va_data_for_EC = pred_vd  # 
        te_data_for_EC = pred_td  # 
        
        # Metrics
        # metrics before error correction
        print('********* Metrics before error correction ***********')
        if fb is False:
            print('MAE metrics on train data, no fb:',mean_absolute_error(np.transpose(cf[:,:-window_len]),np.transpose(pred_d[:,:-window_len]), multioutput='raw_values'))
            print('MAE metrics on valid data:',mean_absolute_error(np.transpose(cf_v[:,:-window_len]),np.transpose(pred_vd[:,:-window_len]), multioutput='raw_values'))
            print('MAE metrics on test data:',mean_absolute_error(np.transpose(cf_t[:,:-window_len]),np.transpose(pred_td[:,:-window_len]), multioutput='raw_values'))

            print('R2 metrics on train data:',r2_score(cf[:,:-window_len],pred_d[:,:-window_len]))
            print('R2 metrics on valid data:',r2_score(cf_v[:,:-window_len],pred_vd[:,:-window_len]))
            print('R2 metrics on test data:',r2_score(cf_t[:,:-window_len],pred_td[:,:-window_len]))
        else:
            max_w = 300 
            print('MAE metrics on train data:',mean_absolute_error(np.transpose(cf[:,:max_w]),np.transpose(pred_d[:,:max_w]),multioutput='raw_values'))
            print('MAE metrics on valid data:',mean_absolute_error(np.transpose(cf_v[:,:max_w]),np.transpose(pred_vd[:,:max_w]), multioutput='raw_values'))
            print('MAE metrics on test data:',mean_absolute_error(np.transpose(cf_t[:,:max_w]),np.transpose(pred_td[:,:max_w]), multioutput='raw_values'))

            print('R2 metrics on train data:',r2_score(cf[:,:max_w],pred_d[:,:max_w]))
            print('R2 metrics on valid data:',r2_score(cf_v[:,:max_w],pred_vd[:,:max_w]))
            print('R2 metrics on test data:',r2_score(cf_t[:,:max_w],pred_td[:,:max_w]))
        n_max = 1 
        for n in range(n_max): # evaluate the error model n_max times and update the predicted output
            # predict the error for validation data
            pred_e, pred_ve, pred_te = fit_error_model(cf,cf_v, cf_t, cf_e, cf_ve,window_len)

            #scaling back the error data
            if e_scale:
                pred_e = np.transpose(preproc_input1.inverse_transform(np.transpose(pred_e)))            
                pred_ve = np.transpose(preproc_input1.inverse_transform(np.transpose(pred_ve)))            
                pred_te = np.transpose(preproc_input1.inverse_transform(np.transpose(pred_te)))            
 
            #Add the predicted error to the predicted value
            pred = pred_e + tr_data_for_EC # training
            pred_v = pred_ve + va_data_for_EC # validation
            pred_t = pred_te + te_data_for_EC # test

            # cal the error from the first round of prediction - (error = truth-predicted)
            cf_pe = cf - pred
            cf_pve = cf_v - pred_v
            cf_pte = cf_t - pred_t
            save_error(cf_pe, model,  fb, 'train_after_EC')
            save_error(cf_pve, model, fb,  'valid_after_EC')
            save_error(cf_pte, model, fb, 'test_after_EC')
             
            # Metrics
            print('********* Metrics after error correction n = ', n+1, '****************')
            if fb is False:
                print('MAE metrics on train data:',mean_absolute_error(np.transpose(cf[:,:-window_len]),np.transpose(pred[:,:-window_len]), multioutput='raw_values'))
                print('MAE metrics on valid data:',mean_absolute_error(np.transpose(cf_v[:,:-window_len]),np.transpose(pred_v[:,:-window_len]), multioutput='raw_values'))
                print('MAE metrics on test data:',mean_absolute_error(np.transpose(cf_t[:,:-window_len]),np.transpose(pred_t[:,:-window_len]), multioutput='raw_values'))

                print('R2 metrics on error corrected train data:',r2_score(cf[:,:-window_len],pred[:,:-window_len]))
                print('R2 metrics on error corrected valid data:',r2_score(cf_v[:,:-window_len],pred_v[:,:-window_len]))
                print('R2 metrics on error corrected test data:',r2_score(cf_t[:,:-window_len],pred_t[:,:-window_len]))
            else:
                print('MAE metrics on train data:',mean_absolute_error(np.transpose(cf[:,:max_w]),np.transpose(pred[:,:max_w]),multioutput='raw_values'))
                print('MAE metrics on valid data:',mean_absolute_error(np.transpose(cf_v[:,:max_w]),np.transpose(pred_v[:,:max_w]), multioutput='raw_values'))
                print('MAE metrics on test data:',mean_absolute_error(np.transpose(cf_t[:,:max_w]),np.transpose(pred_t[:,:max_w]), multioutput='raw_values'))

                print('R2 metrics on error corrected train data:',r2_score(cf[:,:max_w],pred[:,:max_w]))
                print('R2 metrics on error corrected valid data:',r2_score(cf_v[:,:max_w],pred_v[:,:max_w]))
                print('R2 metrics on error corrected test data:',r2_score(cf_t[:,:max_w],pred_t[:,:max_w]))
            
            tr_data_for_EC = pred
            va_data_for_EC = pred_v
            te_data_for_EC = pred_t
        
        # Rescale and save
        # Train
        cf = np.transpose(preproc_input.inverse_transform(np.transpose(cf)))
        pred = np.transpose(preproc_input.inverse_transform(np.transpose(pred)))
        pred_d = np.transpose(preproc_input.inverse_transform(np.transpose(pred_d)))
        np.save('./Coefficients/Truth_train.npy',cf)
        if fb is False:
            np.save('./Coefficients/Prediction_'+model+'_train.npy',pred_d)
            np.save('./Coefficients/Prediction_'+model+'_EC_train.npy',pred)
        else:
            np.save('./Coefficients/Prediction_fb_'+model+'_train.npy',pred_d)
            np.save('./Coefficients/Prediction_fb_'+model+'_EC_train.npy',pred)

        # Valid
        cf_v = np.transpose(preproc_input.inverse_transform(np.transpose(cf_v)))
        pred_v = np.transpose(preproc_input.inverse_transform(np.transpose(pred_v)))
        pred_vd = np.transpose(preproc_input.inverse_transform(np.transpose(pred_vd)))
        np.save('./Coefficients/Truth_valid.npy',cf_v)
        if fb is False:
            np.save('./Coefficients/Prediction_'+model+'_valid.npy',pred_vd)
            np.save('./Coefficients/Prediction_'+model+'_EC_valid.npy',pred_v)
        else:
            np.save('./Coefficients/Prediction_fb_'+model+'_valid.npy',pred_vd)
            np.save('./Coefficients/Prediction_fb_'+model+'_EC_valid.npy',pred_v)


        # Test
        cf_t = np.transpose(preproc_input.inverse_transform(np.transpose(cf_t)))
        pred_t = np.transpose(preproc_input.inverse_transform(np.transpose(pred_t)))
        pred_td = np.transpose(preproc_input.inverse_transform(np.transpose(pred_td)))
        np.save('./Coefficients/Truth_test.npy',cf_t)
        if fb is False:
            np.save('./Coefficients/Prediction_'+model+'_test.npy',pred_td)
            np.save('./Coefficients/Prediction_'+model+'_EC_test.npy',pred_t)
        else:
            np.save('./Coefficients/Prediction_fb_'+model+'_test.npy',pred_td)
            np.save('./Coefficients/Prediction_fb_'+model+'_EC_test.npy',pred_t)

        # Visualize train
        visualize_predictions(pred_d,cf,smean,phi,'train', label_t='POD-'+model, cf_ec_pred=pred)
        visualize_error(cf_e, pred_e, model+'_train') 
        # Visualize valid
        visualize_predictions(pred_vd,cf_v,smean,phi,'valid', label_t='POD-'+model, cf_ec_pred=pred_v)
        visualize_error(cf_ve, pred_ve, model+'_valid') 
        
        # Visualize and analyze test
        visualize_predictions(pred_td,cf_t,smean,phi,'test', label_t='POD-'+model, cf_ec_pred=pred_t)
        visualize_error(cf_te, pred_te, model+'_test') 
        #analyze_predictions(pred_t,cf_t,smean,phi,'test')
