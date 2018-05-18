# StockPredictionLSTM
Predict VN-index trend approach by  RNNs  
 

Step : 
1. Load the dataset from CSV file.
2. Transforming the data to a supervised learning problem.
3. Transforming the data to be stationary.
4. Transforming the data so that it has the scale -1 to 1.
5. Fitting a stateful LSTM network model to the training data.(Use LSTM with loss='mean_squared_error', optimizer='adam’ , batch_size = 1 ,nb_epoch = 3 , neurons = 2)
6. Evaluating the static LSTM model on the test data.
7. Report the performance of the forecasts.
8. Repeat 20 times (time = 406.21 s ) and use mean_square_error to evaluate : 

		1) Test RMSE: 5.574
		2) Test RMSE: 5.593
		3) Test RMSE: 5.610
		4) Test RMSE: 5.592
		5) Test RMSE: 5.639
		6) Test RMSE: 5.609
		7) Test RMSE: 5.648
		8) Test RMSE: 5.619
		9) Test RMSE: 5.632
		10) Test RMSE: 5.638
		11) Test RMSE: 5.627
		12) Test RMSE: 5.630
		13) Test RMSE: 5.618
		14) Test RMSE: 5.640
		15) Test RMSE: 5.587
		16) Test RMSE: 5.635
		17) Test RMSE: 5.585
		18) Test RMSE: 5.575
		19) Test RMSE: 5.640
		20) Test RMSE: 5.580



		count  : 20.00000
		mean   : 5.613534
		std    : 0.024710
		min    : 5.574390
		25%    : 5.590972
		50%    : 5.618729
		75%    : 5.635545
		max    : 5.647541


