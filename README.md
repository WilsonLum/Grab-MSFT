# Grab Challenge - Traffic Management

## Which areas have high / low traffic demand?
- As reference to the data and visualisation below the using max count and highest demand = 1,  latlong(-5.353088,90.818481) is the area having the highest traffic
- Lowest demand area is latlong(-5.24322509765625, 90.8184814453125)

![High Demand Count Clustering](/diagram/geohash__High_demand_Count_clustering.png)

![High Demand COunt Clustering](/diagram/geohash_High_demand_clustering.png)

## Forecast the travel demand for next 15min / 1hour (RMSE)
I'll using minimum approach to feature engineering since I'll be only considering LSTM & Bi-LSTM neural networks which are capable of learning features automatically. I have used few scenario as in using different t-x & t+y , eg: t-3,t-2,t-1 & t to predit t+1 & t+2. Below table is the summary of the best result I could produced.

Due to time factor, I have not be available to relook into the dataset to see if there is other features that I can extract to improve the results. I believe we can add in more features to improve the results further. 

Some observation as follow:
 - It seems that LSTM produced better results than BiLSTM 
 - Deep layers or more neurons may not increase the accuracy after LSTM3.
 - Using further previous state of inout increase the accuracy as comparing using 6 previous state (6in2out) and 3 previous state (3in2out)
 - Best result is LSTM3-6in2out wtih 12,621,82 trainable parameters

**Epoch = 20 using t-2,t-1,t to predict t+1 & t+2**

| Model   |trainable parameters |   train RMSE   ||    val RMSE    ||    test RMSE   ||
| :---------------:|:----------:| :-----:| :-----:| :-----:| :-----:| :-----:| :-----:|
|                  |            |   t+1  |   t+2  |   t+1  |   t+2  |   t+1  |   t+2  |
| LSTM1-3in2out    | 18,453,378 | 0.0693 | 0.0670 | 0.0112 | 0.0185 | 0.0381 | 0.0427 |
| LSTM2-3in2out    | 1,187,586  | 0.0669 | 0.0618 | 0.0104 | 0.0029 | 0.0369 | 0.0432 |
| LSTM3-3in2out    | 12,621,826 | 0.0469 | 0.0394 | 0.0117 | 0.0080 | 0.0322 | 0.0499 |
|                  |            |        |        |        |
| Bi-LSTM1-3in2out | 53,786,114 | 0.1092 | 0.1134 | 0.0070 | 0.0036 | 0.0551 | 0.0627 |
| Bi-LSTM2-3in2out | 3,423,746  | 0.1228 | 0.1234 | 0.0198 | 0.0238 | 0.0602 | 0.0660 |

**Epoch = 20 using t-5,t-4,t-3,t-2,t-1,t to predict t+1 & t+2**

| Model | trainable parameters  |   train RMSE   ||    val RMSE    ||    test RMSE   ||
| :----------------|:----------:| :-----:| :-----:| :-----:| :-----:| :-----:| :-----:|
| LSTM1-6in2out    | 18,453,378 | 0.0591 | 0.0626 | 0.0099 | 0.0071 | 0.0249 | 0.0440 |
| LSTM2-6in2out    | 1,187,586  | 0.0664 | 0.0671 | 0.0197 | 0.0184 | 0.0430 | 0.0468 |
|**LSTM3-6in2out** | 12,621,826 | 0.0215 | 0.0226 | 0.0074 | 0.0071 | 0.0532 | **0.0099**|
|                  |            |        |        |        |        |        |        |
| Bi-LSTM1-6in2out | 53,786,114 | 0.1317 | 0.1348 | 0.0854 | 0.0952 | 0.0957 | 0.1188 |
| Bi-LSTM2-6in2out | 3,423,746  | 0.1195 | 0.1197 | 0.0488 | 0.0547 | 0.0811 | 0.1016 |

