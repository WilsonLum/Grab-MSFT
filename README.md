# Grab Challenge - Traffic Management

## Which areas have high / low traffic demand?
- As reference to the data and visualisation below the using max count and highest demand = 1,  latlong(-5.353088,90.818481) is the area having the highest traffic
- Lowest demand area is latlong(-5.24322509765625, 90.8184814453125)

![High Demand Count Clustering](/diagram/geohash__High_demand_Count_clustering.png)

![High Demand COunt Clustering](/diagram/geohash_High_demand_clustering.png)

## Forecast the travel demand for next 15min / 1hour (RMSE)
I'll using minimum approach to feature engineering since I'll be only considering LSTM & CNN neural networks which are capable of learning features automatically. I have used few scenario as in using different t-x & t+y , eg: t-3,t-2,t-1 & t to predit t+1 & t+2. Below table is the summary of the best result I could produced.

Due to time factor, I have not be available to relook into the dataset to see if there is other features that I can extract to improve the results. I believe we can add in more features to improve the results further. 

It seems that LSTM or BiLSTM have its limit no matter how many deep layers or neurons I added and the best is using lesser layers and neurons as in the model LSTM2-6in2out_relu with trainable parameters of 1,187,586.

**Epoch = 30 using t-2,t-1,t to predict t+1 & t+2**

| Model | trainable parameters  | train RMSE | val RMSE | test RMSE |
| :------------ |:---------------:| :-----:| :-----:| :-----:|
|                     |            |   t+1  |   t+2  |   t+1  |   t+2  |   t+1  |   t+2  |
| LSTM1-3in2out_tanh  | 18,453,378 | 0.1716 | 0.2064 | 0.1054 | 0.1716 | 0.2064 | 0.1054 |
| LSTM2-3in2out_tanh  | 1,187,586  | 0.1726 | 0.2080 | 0.1062 | 0.1716 | 0.2064 | 0.1054 |
| LSTM3-3in2out_tanh  | 12,621,826 | 0.1687 | 0.2040 | 0.1036 | 0.1716 | 0.2064 | 0.1054 |
| LSTM1-3in2out_relu  | 18,453,378 | 0.1679 | 0.2040 | 0.1047 | 0.1716 | 0.2064 | 0.1054 |
| LSTM2-3in2out_relu  | 1,187,586  | 0.1689 | 0.2055 | 0.1052 | 0.1716 | 0.2064 | 0.1054 |
| LSTM3-3in2out_relu  | 12,621,826 | 0.1702 | 0.2057 | 0.1050 | 0.1716 | 0.2064 | 0.1054 |
|                     |            |        |        |        |
| Bi-LSTM1-3in2out_tanh  | 53,786,114 | 0.1615 | 0.1950 | 0.1181 | 0.1716 | 0.2064 | 0.1054 |
| Bi-LSTM2-3in2out_tanh  | 3,423,746  | 0.1601 | 0.1948 | 0.1156 | 0.1716 | 0.2064 | 0.1054 |
| Bi-LSTM1-3in2out_relu  | 53,786,114 | 0.1605 | 0.1946 | 0.1154 | 0.1716 | 0.2064 | 0.1054 |
| Bi-LSTM2-3in2out_relu  | 3,423,746  | 0.1601 | 0.1945 | 0.1134 | 0.1716 | 0.2064 | 0.1054 |

**Epoch = 30 using t-5,t-4,t-3,t-2,t-1,t to predict t+1 & t+2**

| Model | trainable parameters  | train RMSE | val RMSE | test RMSE |
| :------------ |:---------------:| :-----:| :-----:| :-----:|
| LSTM1-6in2out_tanh  | 18,453,378 | 0.1707 | 0.2062 | 0.1053 |
| LSTM2-6in2out_tanh  | 1,187,586  | 0.1692 | 0.2047 | 0.1033 |
| LSTM3-6in2out_tanh  | 12,621,826 | 0.1689 | 0.2046 | 0.1050 |
| LSTM1-6in2out_relu  | 18,453,378 | 0.1715 | 0.2069 | 0.1058 |
| LSTM2-6in2out_relu  | 1,187,586  | 0.1644 | 0.1990 | 0.1032 |
| LSTM3-6in2out_relu  | 12,621,826 | 0.1669 | 0.2028 | 0.1034 |
|                     |            |        |        |        |
| Bi-LSTM1-6in2out_tanh  | 53,786,114 | 0.1618 | 0.1954 | 0.1126 |
| Bi-LSTM2-6in2out_tanh  | 3,423,746  | 0.1617 | 0.1954 | 0.1118 |
| Bi-LSTM1-6in2out_relu  | 53,786,114 | 0.1619 | 0.1954 | 0.1135 |
| Bi-LSTM2-6in2out_relu  | 3,423,746  | 0.1616 | 0.1957 | 0.1095 |

