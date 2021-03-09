# DayMetForecasting.github.io
Time series forecasting for DayMet data

This is a comparative study of different supervised machine learning  time-series  methods for short-term  and  long-term  temperature forecasts on a real world dataset for the daily maximum temperature over North America given by DayMET. DayMET showcases astochastic and high-dimensional spatio-temporal structure and is available at exceptionally fine resolution (a 1 km grid). We apply projection-based reduced order modeling to compress this high dimensional data, while preserving its spatio-temporal structure. We use variants of time-series specific neural network models on this reduced representation to perform multi-step weather predictions. We also use a Gaussian-process based error correction model to improve the forecasts from the neural network models. From our study, we learn that the recurrent neural network based techniques can accurately perform both short-term as well as long-term forecasts, with minimal computational cost as compared to the temoral convolution based techniques. We see that the simple kernel based Gaussian-processes  can  also  predict  the  neural  network  model  errors, which can then be used to improve the long term forecasts.

To run the code, place the data under the data folder. Please email me if you require the data. 

Run the follow commands - 

For training : python ROM_gen.py --win <window_len> --modes <no_of_modes> --epochs <no_of_epochs> --fb --model <model_to_be_applied> --train

For deployment : The above command without --train

example : 
with feedback 

python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000 --fb --model lstm --train

python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000 --fb --model lstm

without feedback

python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000 --model lstm --train

python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000 --model lstm
