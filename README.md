# Supervised Text Classification Challenge

The goal is to improve the category classification performance for a set of text posts. The evaluation metric is the **macro F1 score**. A baseline method based on scikit-learn TfidfVectorizer and LogisticRegression was provided. The baseline score is about **0.3237**. 

The macro F1 score is given by the average of the F1 scores for each category (label), computed independently from their size. Therefore, it may not be the best metric to characterize the performance of a classifier in case of unbalanced class sizes. 

I explored various features, classifiers and hyperparameter settings. The final classifier is a majority vote ensemble of 2 logistic regression classifiers (on different set of features) and 1 gradient boosted tree classifier (of the XGBoost library) classifier leading to a final macro F1 score equal to **0.5574** ) Those are the main takeaways: 
1. Overall, logistic regression outperformed the other classifiers I tried (e.g. gradient boosted trees and random forest).
2. Gradient tree boosting of the XGBoost library was the 2nd best classifier after logistic regression.
3. Performing stemming (with preliminary stop word removal) improved the performance of logistic regression and XGBoost.
4. The addition of topic features via Latent Dirichlet Allocation improved slightly the performance for two of the classifiers.
5. The manually engineered features that gave a slight improvement to one of the classifiers are the length of a post (by number of tokens) and the question mark ('?') count.

The outcome (1) is not surprising, because the number of training samples (~ 14,000) is relatively low with respect to the number of features (e.g. ~ 10,000 extracted via scikit-learn TfidfVectorizer) for tree based methods to be fully effective. In this case a simple linear classifier is expected to do better than non-linear classifiers. I would expect tree based methods to outperform logistic regression if *number of samples* $\gg$ *number of features*. See for example the discussion on (http://fastml.com/classifying-text-with-bag-of-words-a-tutorial/).


### The Dataset
The dataset consist in 14048 text posts for training and 3599 for testing. There are 17 topic categories: personal, meetup, misc, relationships etc. The topic sizes are quite unbalanced. The proportions of categories in training and test sets seem to match.

The deatils about features, classifiers and hyperparameters are given in the remaining of this notebook.

**Author:** Lorenzo Rossi (lorenzo.rossi@gmail.com)
