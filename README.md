# Bank Loan Prediction
## Last Updated: December 8, 2022

## 1 Introduction

For this competition we had to predict whether a person would repay their loan. In the initial data,
we got many different pieces of information like annual income, employment, race, etc. (As it is illegal
to use race to make loan decisions, it was immediately removed.) Furthermore, theytrainis the
loanpaidcolumn. There were over a million data points to train on. However, this data set requires
a lot of pre-processing as you will see soon. Furthermore, I use a gradient boosted decision tree to
achieve an accuracy of over 80.5%.

## 2 Pre-processing

To pre-process the data, many steps were taken. Here are the major things:

1) First, bad columns like race, ID, and extended reason were removed.
2) Then, rows with lots of missing values were removed as they are bad to learn.
3) Rows with few missing values had their missing values filled in with mode or median (based on
categorical or numerical) of that column.
4) Next, new columns that provided more prediction power through having a high correlation with
loan paid were added. For example, income to loan amount ratio was added.
5) As an extension of the previous step, I removed ficoscorerangehigh and ficoscorerangelow as they were very highly correlated. I still kept the information provided by this column by adding a new column with the average of high and low. Furthermore, I added another column that contained the range between the high and low.
6) Next, I converted the loanduration and employmentlength columns from strings to corresponding integers.
7) I then encoded ordinal columns. To find what each value should be encoded with, I performed t-tests.
8) For columns with lots of unique values, I simplified them by keeping only the 5 most common set all other values to other.
9) To encode the states column, one-hot encoding was not a viable option as there are a lot of unique values. To get around this, I gave them a number between 1 and 50 based on the average. This may cause what is referred to as a data leakage problem, but I was not sure how else to encode this column.
10) The zip code columns also required special encoding as it had lots of unique values. To encode here, I grouped zip codes based on the loan paid average. There were 5 categories: bad, mid bad, mid, mid good, and good.
11) To complete encoding, I performed one hot encoding on any remaining categorical columns.
12) An interesting thing I noticed when exploring the data was with the monthssincelastdelinq column. When someone did not have any delinquency, it seemed that they had a value of 0. Since this was ordinally wrong, I changed all 0s to be the new maximum value, which was much greater than the original max.
13) Finally, I normalized all the columns that I through were appropriate.

Note: When doing the pre-proccessing for the validation dataset, I made sure to not remove any columns with missing values. Furthermore, I filled in the missing values based on the mode and median of the training dataset. Furthermore, I normalized based on the mean and standard deviation of the training dataset as well to maintain consistency.

## 3 Knowledge Representation

The model I used for this competition was the Gradient Boosted Classifier from the Sklearn library.
This is essentially a version of boosted decision trees as after each iteration, it emphasizes the data
points that it got wrong for the following iteration. The model space therefore includes all the possible
trees each with a different node split. Since this model also optimizes the weights of the leaf nodes,
really, the model space contains all unique set of trees (both in terms of node split decisions and leaf
node weights). Generally though, these trees are stumps (small max depth) as boosting is meant for
weak learners. The Gradient Boosted Classifier works specifically by first building out the optimal set
of trees using algorithms like CART or C4.5. Then, it finds the best set of lead node weights to further
improve its prediction power.

## 4 Score Function

The general version of the score function used by the Gradient Boosted Classifier is the following:
Score(k
thiteration)
= ΣJj=1(Σi∈Ij(h(ik)+α∗wj)α∗wj) + Ω(α∗fk)
Here, the h term is the residual between the correct and predicted y values. Then, the Omega term
is for regularization. J is the number of leaves in the tree of thekthiteration. I represents the data
points that end up injthleaf. Next, thewjrepresents the weights of thejthleaf node. Finally,αis the
learning rate/contribution rate of each tree. This score function helps the model both building out the
trees and optimize the weights of the leaf nodes. This is because the h term represents how good the
tree is without the weights. Then the weights are presented by theα∗wjterm. Furthermore, to pre-
vent weights from making the model too complex, there is a regularization term containing the weights.

**Domain adaptation is a work in progress.**

## 5 Search Method

The search algorithm (could be CART or C4.5) to build the trees is a greedy one that makes the locally
optimal choice that minimizes entropy (could be using Information or Gini gain). Keep in mind, when
making the trees, boosting is also occurring. The way this works is that for each iteration, the model
focuses on the data points that the previous iterations got wrong. This happens through the weights
of each data point changing based on whether or not they were properly predicted. Furthermore,αis
determined by the number of correct predictions made. Essentially, the same as building any regular
decision tree. For optimizing the weights part, the search algorithm is the gradient descent (since
trying to minimize). It utilizes the gradient of the score function. The general version of this gradient
is the following:
= ΣJj=1(∂w∂j(Σi∈Ij(h(ik)+α∗wj))α∗wj) +∇(Ω(α∗fk))

So overall, the model is boosting decision trees and then adding gradient descent in order to optimize
the weights.

## Issues
K-means clustering creating clusters with size 1 even if # of clusters is not set to (n - 1).
