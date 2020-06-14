# Grab-MSFT
## Grab Challenge


**Epoch = 30 using t-2,t-1,t to predict t+1 & t+2**

| Model | trainable parameters  | train RMSE | val RMSE | test RMSE |
| :------------ |:---------------:| :-----:| :-----:| :-----:|
| LSTM1-3in2out_tanh  | 18,453,378 | 0.1716 | 0.2064 | 0.1054 |
| LSTM2-3in2out_tanh  | 1,187,586  | 0.1726 | 0.2080 | 0.1062 |
| LSTM3-3in2out_tanh  | 12,621,826 | 0.1687 | 0.2040 | 0.1036 |
| LSTM1-3in2out_relu  | 18,453,378 | 0.1679 | 0.2040 | 0.1047 |
| LSTM2-3in2out_relu  | 1,187,586  | 0.1689 | 0.2055 | 0.1052 |
| LSTM3-3in2out_relu  | 12,621,826 | 0.1702 | 0.2057 | 0.1050 |

**Epoch = 30 using t-5,t-4,t-3,t-2,t-1,t to predict t+1 & t+2**
