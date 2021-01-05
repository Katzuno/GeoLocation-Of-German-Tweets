# GeoLocation-Of-German-Tweets
Project 1 Practical Machine Learning - Geolocation of German Tweets

At the data preprocessing part I got the best score by preprocessing the datas using TF-IDF algorithm, implemented by the Tokenizer class found in keras.preprocessing library.
I chose to use the TF - IDF algorithm after I tried multiple embedding options like Doc2Vec and Bag of Words.
As I expected, the TF - IDF (Term Frequency - Inverse Document Frequency) improved the results by a good margin, because it calculates how important a word is to a document in the collection, by calculating the multiple of the two metrics: term frequency and inverse document frequency.
	I fitted the tokenizer on texts from all datasets, including the ones from training, validation and prediction datasets because I realised that this helps the model to deliver more accurate predictions, because it had a complete vocabulary of the german language.
	As part of the tokenizer class I removed the punctuation and turned phrases into words, by splitting by the ‘ ‘ character, before applying TF-IDF.
	After preprocessing the datasets I got a dataframe for training with shape (22583, 142293) because I splitted all twitter posts into single words.
	To bring features in the same range I normalized my data with the L2 norm. I tried other norms too like L1 or MAX, but L2 scored considerably better that MAX, which score 0.69 on Kaggle private leaderboard and 0.71 on my local machine.
	After scaling the training samples, I trained a Kernel Ridge Regression model with ‘rbf’ kernel and alpha 10^(-5) after searching some parameters configuration with Grid Search.
	To validate the results on my local machine I used the K-Fold Cross Validation procedure from sklearn library, performing 3 splits and I found out that it was very good method because I always scored on local machine almost the same MAE as I scored on Kaggle.
