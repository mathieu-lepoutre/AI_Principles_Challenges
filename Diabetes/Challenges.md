
# Artificial Intelligence Principles - Challenges

32318/1700/1920/1/22
D'Haese David

---

## Diabetes

The goal of this data analysis report is to predict whether and how soon a diabetes patient would be readmitted to the hospital based on demographics as well as visit and medication details.
​
This, time, you'll have to start from (almost) scratch.

### Challenge II.01

The one and only assignment is simple. complete the `Diabetes.ipynb`. This time, you'll make use of a [decision tree classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) and will be working with 3 subsets (training, validation, test) instead of 2. Remember to keep the test data set for the very last! In first instance, you'll need to train the learner using the training data and set the tree depth to a maximum of 5 levels (via the `max_depth` parameter).

One of the attributes of the learner after training is the `feature_importances_` attribute. This attribute contains the contributions of the features in the order as they are provided to the learner. The steps you'll need to take are:

1. Fit the learner using the training data `feat_trai`, `outc_trai`
1. Extract the feature `feature_importances_` into the variable `cont`
1. Find the indices of the features with nonzero contribution, sort them according to contribution (check out the `argsort` function of numpy) and place them in the variable `cont_nonz`
1. Sequentially remove the least contributing feature of the training and validation sets and refit the learner on the remaining features (same depth for now) + use the `score` function on the `learner` object to measure the performance on the validation set
1. Make a plot with on the x axis the decreasing number of features used and on the y-axis the accuracy score
1. From this graph, determine the ideal number of features and refit the learner with this ideal feature set

Under the heading `Hyperparameters`:

1. Now test the best `max_depth` value by doing a _grid search_ from 1:15
1. Create a plot with on the x axis the `max_depth` and on the y-axis the accuracy score

Under the best model, you need to refit the learner using the best combination of feature set and parameters.

Remember to add a conclusion at the end.
