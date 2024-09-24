# Toxic-Comment-Classification-Challenge

## Intorduction

The simulation of Toxic Comment Classification Challenge by Kaggle is conducted to classify different levels of toxic online comments in social media. Data quality analysis and the data cleaning and preprocessing techniques will first be applied in train.csv, then train the three different machine learning models. The best model will be selected to predict the probability of toxic comment in test.csv.

## Steps on training the models

1.  Data quality checking: 'train.csv' contains 159571 rows of comments and 8 columns of ‘id’, ‘comment_text’ and 6 different classes. It is multilabel classification problem because each row can be assigned to multiple classes simultaneously. No null values is found in the whole dataset. However, dataset imbalanced is detected as 143346 comments out of 159571 are clean (no label in all classes) while only 16225 comments have at least 1 label.
   
2.  Data cleaning and preprocessing: A function is created to clean the text data and remove all the unnecessary elements by the following actions:
   - Replace common short-form words with expanded forms
   - Remove html chars
   - Remove text in square brackets and parenthesis
   - Remove non-ascii chars
   - Remove words containing numbers
   - Tokenize the text
   - Lemmatize each token and remove stop words, punctuation, and whitespace
   - Join the tokens back into a cleaned text

3.  Data preparation: Train test split is applied to split the data into training and testing sets. 80% of the data will be used for training and the rest will be used for prediction. It is worth noting that machine learning models only understand numbers so converting texts into numbers is necessary step. Bag of Words and TF-IDF methods are used for text vectorization by transforming X_train and X_test by applying CountVectorizer and TfidfVectorizer in scikit-learn library.

4.  Model training: 3 ML models is deployed after consideration of dataset imbalanced and model efficiency.

   - ComplementNB - one of the naive bayes algorithms but it is particularly used for imbalanced dataset.
   - RandomForestClassifier - set of decision trees, class_weight='balanced' parameter is added to assign higher weights to the minority class (less frequent class) and lower weights to the majority class (more frequent class). 
   - SGDClassifier - it is equivalent to LogisticRegressClassifier if the parameter of loss='log_loss' is added but SGDClassifier offers higher efficiency of handling large dataset. class_weight='balanced' parameter is also added to address da-taset imbalanced issue.

5.  Model evaluation: Given the results provided, the choice of feature extraction methods (Bag of words and TFIDF) did not have significant impact of the model performance in terms of accuracy. However, models with TfidfVectorizer performed better with weighted average f1 score. Considering the imbalanced dataset, weighted average f1 score is more reliable metrics as it considers both precision and recall simultaneously. SGDClassifier with TfidfVectorizer got the highest weighted average f1 score among the models despite the slightly lower accuracy so clf6 is considered the best models to classify text data in test.csv.

## Kaggle Submission

Repeat the steps mentioned in part 2 in test.csv. Then predict the probability of each comment using GDClassifier with TfidfVectorizer, export the result to csv file and submit to Kaggle. The score of my best model is 0.97276 (public score). Result in Kaggle was decent compared to top performers. Further studies are required such as hyperparameter tuning and imbalanced dataset handling to improve the result.
