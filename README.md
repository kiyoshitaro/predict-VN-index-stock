# predict-VN-index-stock
Predict VN-index trend approached by RNNs(LSTM)


Step : 
1. Load the dataset from CSV file.
2. Transforming the data to a supervised learning problem.
3. Transforming the data to be stationary.
4. Transforming the data so that it has the scale -1 to 1.
5. Fitting a stateful LSTM network model to the training data.(Use LSTM with loss='mean_squared_error', optimizer='adam’ , batch_size = 1 ,nb_epoch = 20 , neurons = 4 )
6. Evaluating the static LSTM model on the test data.
7. Report the performance of the forecasts.
8. Repeat 10 times (time = 602.65 s ) and use mean_square_error to evaluate : 

            1) Test RMSE: 4.312
            2) Test RMSE: 4.322
            3) Test RMSE: 4.318
            4) Test RMSE: 4.306
            5) Test RMSE: 4.339
            6) Test RMSE: 4.310
            7) Test RMSE: 4.323
            8) Test RMSE: 4.319
            9) Test RMSE: 4.311
            10) Test RMSE: 4.325
            
            
            rmse
            count  10.000000
            mean    4.318514
            std     0.009564
            min     4.306025
            25%     4.311294
            50%     4.318286
            75%     4.322575
            max     4.339062


