from tensorflow.keras.layers import Input, Dense, Lambda, Add, LSTM, dot, concatenate, Activation, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten, RepeatVector, TimeDistributed, SpatialDropout1D
from tensorflow.keras.models import load_model, Sequential, Model
from tcn import TCN, tcn_full_summary

def gen_lstm(seq_num,n_features, dropout, training_t):
    lstm_inputs = Input(shape=(seq_num, n_features,))
    l1 = LSTM(50,return_sequences=True)(lstm_inputs)
    l1 = Dropout(0.1)(l1,training=True)
    l1 = LSTM(50,return_sequences=True)(l1)
    l1 = Dropout(0.1)(l1,training=True)
    op = Dense(n_features, activation='linear', name='output')(l1)
    model = Model(inputs=lstm_inputs, outputs=op)
    return model

def gen_bilstm(seq_num,n_features, dropout, training_t):
    blstm_inputs = Input(shape=(seq_num, n_features,))
    l1 = Bidirectional(LSTM(50,return_sequences=True))(blstm_inputs)
    l1 = Dropout(0.1)(l1,training=training_t)
    l1 = Bidirectional(LSTM(50,return_sequences=True))(l1)
    l1 = Dropout(0.1)(l1,training=training_t)
    op = Dense(n_features, activation='linear', name='output')(l1)
    model = Model(inputs=blstm_inputs, outputs=op)
    return model

def gen_1dconv(seq_num,n_features, dropout, training_t):
    conv_inputs = Input(shape=(seq_num, n_features))
    cnn1 = Conv1D(filters=64, kernel_size=2, activation='relu')(conv_inputs) # o = 6,64
    cnn1 = SpatialDropout1D(rate=0.1)(cnn1, training=training_t)
    cnn1 = MaxPooling1D(pool_size=2)(cnn1) #3,64
    cnn1 = Flatten()(cnn1)
    cnn1 = Dense(50, activation='relu')(cnn1)
    op = Dense(seq_num * n_features)(cnn1)
    model = Model(inputs=conv_inputs, outputs=op)
    return model

def gen_tcn(seq_num,n_features, dropout, training_t): 
    tcn_inputs = Input(batch_shape=(None,seq_num, n_features))
    tcn = TCN(nb_filters=64, kernel_size=2, activation='relu', dropout_rate=0.1, return_sequences=False)(tcn_inputs, training=training_t) # o = 6,64
    op = Dense(seq_num * n_features)(tcn)
    model = Model(inputs=tcn_inputs, outputs=op)
    return model

def gen_stcn(seq_num,n_features, dropout, training_t):
    tcn_inputs = Input(batch_shape=(None,seq_num, n_features))
    tcn = TCN(nb_filters=64, kernel_size=2, activation='relu', dropout_rate=0.1, return_sequences=True)(tcn_inputs, training=training_t) # o = 6,64
    tcn = TCN(nb_filters=64, kernel_size=2, activation='relu', dropout_rate=0.1, return_sequences=False)(tcn, training=training_t) # o = 6,64
    op = Dense(seq_num * n_features)(tcn)
    model = Model(inputs=tcn_inputs, outputs=op)
    return model

def gen_EncDec(seq_num,n_features, dropout, training_t):    
    enc_inputs = Input(shape=(seq_num, n_features))
    enc_l1 = LSTM(100,return_state=True)
    #enc_l1 = Dropout(0.1)(enc_l1, training=training_t)
    enc_outputs = enc_l1(enc_inputs)
    encoder_states1 = enc_outputs[1:]
    decoder_inputs = RepeatVector(seq_num)(enc_outputs[0])
    dec_l1 = LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
    dec_l1 = Dropout(0.1)(dec_l1, training=training_t)
    dec_outputs = TimeDistributed(Dense(n_features))(dec_l1)
    model = Model(inputs=enc_inputs, outputs=dec_outputs)
    return model

def gen_EncDec2(seq_num,n_features, dropout, training_t):    
    enc_inputs = Input(shape=(seq_num, n_features))
    enc_l1 = LSTM(100,return_sequences = True,return_state=True)
    enc_outputs1 = enc_l1(enc_inputs)
    encoder_states1 = enc_outputs1[1:]
    enc_l2 = LSTM(100, return_state=True)
    enc_outputs2 = enc_l2(enc_outputs1[0])
    encoder_states2 = enc_outputs2[1:]
    decoder_inputs = RepeatVector(seq_num)(enc_outputs2[0])
    dec_l1 = LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
    dec_l2 = LSTM(100, return_sequences=True)(dec_l1,initial_state = encoder_states2)
    dec_outputs2 = TimeDistributed(Dense(n_features))(dec_l2)
    model = Model(inputs=enc_inputs, outputs=dec_outputs2)
    return model
model_func = {'lstm':gen_lstm, 'blstm':gen_bilstm, '1dconv':gen_1dconv, 'tcn':gen_tcn, 'stcn':gen_stcn, 'ED':gen_EncDec, 'ED2':gen_EncDec2}
def get_model(model_name, seq_n, n_features, dropout, training_t):
    return model_func[model_name](seq_n, n_features, dropout, training_t)
