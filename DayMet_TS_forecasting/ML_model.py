from Config import *
import numpy as np
import tensorflow as tf
from models import get_model
import time
from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import normalize

lwt = np.array([1.7581984e+10, 7.0274842e+08, 4.9741226e+08, 3.3224438e+08, 3.1912090e+08])
lwts = normalize(lwt[:,np.newaxis], axis=0).ravel()
repeat_m =  True 
dropout_t = True
mod_n = ''
class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        # use this value as reference to calculate cummulative time taken
        self.timetaken = time.clock()
       
    def on_epoch_end(self,epoch,logs = {}):
        temp_t = time.clock() 
        self.times.append((epoch,temp_t - self.timetaken))
        self.timetaken = temp_t
        
    def on_train_end(self,logs = {}):
        timel = (list(zip(*self.times))[1])
        
        filename =  './Training/'+str(mod_n)+'_'+'epoch_time.txt'  
        with open(filename, 'w+') as f:
            f.write("total number of epochs : {} \n ".format(len(self.times)))
            f.write(str(timel))
            f.write("\n mean epoch training time : {} \n ".format(np.mean(timel)))
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# architecture
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def model_for_dynamics(cf_trunc,cf_trunc_v,num_epochs,seq_num,train_mode, model_name):
    global mod_n
    features = np.transpose(cf_trunc)
    features_v = np.transpose(cf_trunc_v) # Validation
    states = np.copy(features[:,:]) #Rows are time, Columns are state values
    states_v = np.copy(features_v[:,:]) #Rows are time, Columns are state values #356, 730
    
    # Need to make batches of 10 input sequences and 1 output
    # Training
    total_size = np.shape(features)[0]-2*seq_num + 1 #717
    input_seq = np.zeros(shape=(total_size,seq_num,np.shape(states)[1])) #(717, 7, 730)
    output_seq = np.zeros(shape=(total_size,seq_num,np.shape(states)[1]))


    for t in range(total_size):
        input_seq[t,:,:] = states[None,t:t+seq_num,:]
        output_seq[t,:,:] = states[None,t+seq_num:t+2*seq_num,:]

    # Validation
    total_size = np.shape(features_v)[0]-2*seq_num + 1
    input_seq_v = np.zeros(shape=(total_size,seq_num,np.shape(states_v)[1]))
    output_seq_v = np.zeros(shape=(total_size,seq_num,np.shape(states_v)[1]))

    for t in range(total_size):
        input_seq_v[t,:,:] = states_v[None,t:t+seq_num,:]
        output_seq_v[t,:,:] = states_v[None,t+seq_num:t+2*seq_num,:]

    idx = np.arange(total_size)
    np.random.shuffle(idx)
    
    input_seq = input_seq[idx,:,:]
    output_seq = output_seq[idx,:,:]

    if model_name == '1dconv' or model_name == 'tcn' or model_name == 'stcn':
        n_output = output_seq.shape[1] * output_seq.shape[2]
        output_seq = output_seq.reshape((output_seq.shape[0], n_output))
        output_seq_v = output_seq_v.reshape((output_seq_v.shape[0], n_output))
    
    if dropout_t == True:
        dropout = 0.2
        training_t = True
    else:
        dropout = 0 
        training_t = False
    
    mod_n = model_name
    model = get_model(model_name, seq_num, np.shape(states)[1], dropout, training_t)
    
    # design network
    my_adam = optimizers.Adam(lr=0.0001, decay=0.0)
    filepath = "./Training/best_weights_" + model_name +  ".h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
    timetaken = timecallback()
    callbacks_list = [checkpoint,earlystopping,timetaken]
    print(model.summary())    
    # fit network
    lossWts=None
    model.compile(optimizer=my_adam,loss='mean_squared_error',loss_weights=lossWts, metrics=[new_r2], run_eagerly=True)
    if train_mode:
        train_history = model.fit(input_seq, \
                                output_seq, \
                                epochs=num_epochs, \
                                batch_size=batchsize, \
                                callbacks=callbacks_list, \
                                validation_data=(input_seq_v, output_seq_v))
        #print(f"early stopped epoch number : {earlystopping.stopped_epoch}")

        np.save('./Training/Train_Loss_'+model_name + '.npy',train_history.history['loss'])
        np.save('./Training/Val_Loss_'+model_name + '.npy',train_history.history['val_loss'])

    model.load_weights(filepath)
    return model

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# LSTM architecture for error
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def model_for_error(cf_trunc,cf_trunc_v,cf_e,cf_ve,num_epochs,seq_num,train_mode):

    features = np.transpose(cf_trunc)
    features_v = np.transpose(cf_trunc_v) # Validation
    errors = np.transpose(cf_e)
    errors_v = np.transpose(cf_ve)
    states = np.copy(features[:,:]) #Rows are time, Columns are state values
    states_v = np.copy(features_v[:,:]) #Rows are time, Columns are state values #356, 730
    states_e = np.copy(errors[:,:]) #Rows are time, Columns are state values
    states_ve = np.copy(errors_v[:,:]) #Rows are time, Columns are state values #356, 730
 
    
    # Need to make batches of 10 input sequences and 1 output
    # Training
    total_size = np.shape(features)[0]-2*seq_num + 1 #717
    input_seq = np.zeros(shape=(total_size,seq_num,np.shape(states)[1])) #(717, 7, 730)
    output_seq = np.zeros(shape=(total_size,seq_num,np.shape(states_e)[1]))
    for t in range(total_size):
        input_seq[t,:,:] = states[None,t:t+seq_num,:]
        output_seq[t,:,:] = states_e[None,t+seq_num:t+2*seq_num,:]

    # Validation
    total_size = np.shape(features_v)[0]-2*seq_num + 1
    input_seq_v = np.zeros(shape=(total_size,seq_num,np.shape(states_v)[1]))
    output_seq_v = np.zeros(shape=(total_size,seq_num,np.shape(states_v)[1]))

    for t in range(total_size):
        input_seq_v[t,:,:] = states_v[None,t:t+seq_num,:]
        output_seq_v[t,:,:] = states_ve[None,t+seq_num:t+2*seq_num,:]

    idx = np.arange(total_size)
    np.random.shuffle(idx)
    
    input_seq = input_seq[idx,:,:]
    output_seq = output_seq[idx,:,:]
   
    model = get_model(model_name, seq_num, np.shape(states)[1], None, None)

    # design network
    my_adam = optimizers.Adam(lr=0.0001, decay=0.0)

    filepath = "./Training/best_weights_lstm_errors.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
    callbacks_list = [checkpoint,earlystopping]
    
    # fit network
    model.compile(optimizer=my_adam,loss=Huber(0.2),metrics=[new_r2])
   
    if train_mode:
        train_history = model.fit(input_seq, \
                                output_seq, \
                                epochs=num_epochs, \
                                batch_size=batchsize, \
                                callbacks=callbacks_list, \
                                validation_data=(input_seq_v, output_seq_v))
        
        np.save('./Training/Train_Loss_lstm_errors.npy',train_history.history['loss'])
        np.save('./Training/Val_Loss_lstm_errors.npy',train_history.history['val_loss'])

    model.load_weights(filepath)

    return model

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# model forecast
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def evaluate_rom_deployment(model,dataset,tsteps,num_modes,seq_num, model_name, repeat_m=False):

    if repeat_m == True:
        max_it = 20
    else:
        max_it = 1
    # Make the initial condition from the first seq_num columns of the dataset
    features = np.transpose(dataset)  
    input_states = np.copy(features)

    state_tracker = np.zeros(shape=(max_it,tsteps,np.shape(features)[1]),dtype='double')
    state_tracker[:,0:seq_num,:] = input_states[0:seq_num,:]

    total_size = np.shape(features)[0]-seq_num + 1

    for t in range(seq_num,total_size,seq_num):
        model_input = np.expand_dims(input_states[t-seq_num:t,:],0)
        for r in range(max_it):
            output_state = model.predict(model_input)
            if model_name == '1dconv' or model_name == 'tcn' or model_name == 'stcn':
                output_state = output_state.reshape(1,seq_num,np.shape(features)[1])
         
            state_tracker[r,t:t+seq_num,:] = output_state[0,:,:]
    return np.transpose(output_state), np.transpose(state_tracker[:,:,:])

#-------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# LSTM error forecast
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def evaluate_rom_error(model,dataset,tsteps,num_modes,seq_num):

    # Make the initial condition from the first seq_num columns of the dataset
    features = np.transpose(dataset)  
    input_states = np.copy(features)

    state_tracker = np.zeros(shape=(1,tsteps,np.shape(features)[1]),dtype='double')
   
    total_size = np.shape(features)[0]-seq_num + 1

    for t in range(seq_num,total_size,seq_num):
        model_input = np.expand_dims(input_states[t-seq_num:t,:],0)
        output_state = model.predict(model_input)
        state_tracker[0,t:t+seq_num,:] = output_state[0,:,:]

    return np.transpose(output_state), np.transpose(state_tracker[0,:,:])

#-------------------------------------------------------------------------------------------------
# model forecast with last predicted output
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def evaluate_rom_deployment_predicted_history(model,dataset,tsteps,num_modes,seq_num, model_name, repeat_m=False):
   
    if repeat_m == True:
        max_it = 20
    else:
        max_it = 1
    # Make the initial condition from the first seq_num columns of the dataset
    features = np.transpose(dataset)  
    input_states = np.copy(features)

    state_tracker = np.zeros(shape=(max_it,tsteps,np.shape(features)[1]),dtype='double')
    state_tracker[:,0:seq_num,:] = input_states[0:seq_num,:]

    total_size = np.shape(features)[0]-seq_num + 1

    #initialize the first chunk of true test data
    model_input = np.expand_dims(input_states[0:seq_num,:],0)
    for t in range(seq_num,total_size,seq_num):
        for r in range(max_it):
            output_state = model.predict(model_input)
            if model_name == '1dconv' or model_name == 'tcn' or model_name == 'stcn':
                output_state = output_state.reshape(1,seq_num,np.shape(features)[1])
            
            state_tracker[r,t:t+seq_num,:] = output_state[0,:,:]
        # now use the last predicted data chunk as input to the next predicting window
        if model_name == '1dconv' or model_name == 'tcn' or model_name == 'stcn':
            model_input = output_state
        else:
            model_input = output_state.reshape(1,seq_num,np.shape(features)[1])
       
    return np.transpose(output_state), np.transpose(state_tracker[:,:,:])
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Metrics
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def new_r2(y_true, y_pred):
    SS_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred), axis=0)
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true, axis=0)), axis=0)
    output_scores =  1 - SS_res / (SS_tot + tf.keras.backend.epsilon())
    r2 = tf.keras.backend.mean(output_scores)
    return r2

def coeff_determination(y_pred, y_true): #Order of function inputs is important here        
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
    
