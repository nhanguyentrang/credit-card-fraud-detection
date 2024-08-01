# Credit card fraud detection
Write some description here.

```
## Prevent warning messages from displaying with code results
import warnings
warnings.simplefilter(action = 'ignore')
```


# 1. Data import

```
import pandas as pd  # for working with dataframe
```

```
creditcard = pd.read_csv('creditcard.csv')
creditcard
```

| Time |    V1    |    V2    |    V3    |    V4    |    V5    | ... |    V26   |    V27   |    V28   | Amount | Class |
|-----:|---------:|---------:|---------:|---------:|---------:|-----|---------:|---------:|---------:|-------:|------:|
|     0|       0.0| -1.359807| -0.072781|  2.536347|  1.378155| ... | -0.189115|  0.133558| -0.021053|  149.62|      0|
|     1|       0.0|  1.191857|  0.266151|  0.166480|  0.448154| ... |  0.125895| -0.008983|  0.014724|    2.69|      0|
|     2|       1.0| -1.358354| -1.340163|  1.773209|  0.379780| ... | -0.139097| -0.055353| -0.059752|  378.66|      0|
|     3|       1.0| -0.966272| -0.185226|  1.792993| -0.863291| ... | -0.221929|  0.062723|  0.061458|  123.50|      0|
|     4|       2.0| -1.158233|  0.877737|  1.548718|  0.403034| ... |  0.502292|  0.219422|  0.215153|   69.99|      0|
|   ...|       ...|       ...|       ...|       ...|       ...| ... |       ...|       ...|       ...|     ...|    ...|
|284802|  172786.0|-11.881118| 10.071785| -9.834783| -2.066656| ... |  0.250034|  0.943651|  0.823731|    0.77|      0|
|284803|  172787.0| -0.732789| -0.055080|  2.035030| -0.738589| ... | -0.395255|  0.068472| -0.053527|   24.79|      0|
|284804|  172788.0|  1.919565| -0.301254| -3.249640| -0.557828| ... | -0.087371|  0.004455| -0.026561|   67.88|      0|
|284805|  172788.0| -0.240440|  0.530483|  0.702510|  0.689799| ... |  0.546668|  0.108821|  0.104533|   10.00|      0|
|284806|  172792.0| -0.533413| -0.189733|  0.703337| -0.506271| ... | -0.818267| -0.002415|  0.013649|  217.00|      0|

*284807 rows × 31 columns*


# 2. Data inspection

```
import matplotlib.pyplot as plt   # for plotting
import seaborn as sns             # for plotting
import numpy as np                # for array processing & scientific calculations
from scipy.stats import pearsonr  # for computing p-values
```


## 2.1. Data distribution
### a. Plot PCA features

```
## Create a 4x7 grid of subplots for 28 features (excluding Time, Amount and Class)
fig, axes = plt.subplots(4, 7, figsize = (20, 15))
axes = axes.flatten()  # convert 2D array into 1D array for easier iteration

## List of 28 features to plot
features = [col for col in creditcard.columns if col not in ['Time', 'Amount', 'Class']]

## Plot the distribution of 28 features
for i, ax in enumerate(axes):
    if i < len(features):
        sns.boxplot(x = creditcard[features[i]], ax = ax)
        ax.set_title(features[i])
    else:
        ax.axis('off')  # hide unused subplots

## Show plots
plt.tight_layout()
plt.show()
```

<img width="951" alt="image" src="https://github.com/user-attachments/assets/79e63b0c-f6bc-4d23-b3c6-7ce739d2f291">


### b. Plot Time & Amount features

```
## Create a 1x2 grid of subplots for Time and Amount
fig, axes = plt.subplots(1, 2, figsize = (16, 6))

## Plot Time feature
sns.boxplot(x = creditcard['Time'], ax = axes[0])
axes[0].set_title('Time')

## Plot Amount feature
sns.boxplot(x = creditcard['Amount'], ax = axes[1])
axes[1].set_title('Amount')

## Show plots
plt.tight_layout()
plt.show()
```

<img width="1121" alt="image" src="https://github.com/user-attachments/assets/c488552a-c12a-454b-b729-51fe03aaecd6">


## 2.2. Imbalance ratio

```
## Extract fraud and non-fraud count
fraud_count = np.sum(creditcard['Class'] == 1)
legit_count = np.sum(creditcard['Class'] == 0)

## Data preparation for plotting
trans_types = ['Fraudulent', 'Legitimate']
counts = [fraud_count, legit_count]

## Visualisation
plt.bar(trans_types, counts, color = ['red', 'blue'])
plt.xlabel('Transaction Type')
plt.ylabel('Count')
plt.title('Fraudulent vs Legitimate Transactions')
plt.show()

## Imbalance ratio calculation
imbalance = fraud_count / legit_count * 100
print('Fraud-to-non-fraud ratio = %.2f%%' % imbalance)
```

<img width="598" alt="image" src="https://github.com/user-attachments/assets/466baa5e-956c-4bed-9cbe-24b0f1aeba78">

*Fraud-to-non-fraud ratio = 0.17%*


## 2.3. Multicollinearity

```
## Exclude Class feature
features = creditcard.drop(columns = ['Class'])

## Correlation coefficient matrix
corr_mat = features.corr()

## P-values calculation
# initialise p-value matrix
p_val = pd.DataFrame(np.zeros((features.shape[1], features.shape[1])),
                     columns = features.columns, index = features.columns)
# calculate p-values
for row in features.columns:
    for col in features.columns:
        if row != col:
            p_val.loc[row, col] = pearsonr(features[row], features[col])[1]
        else:
            p_val.loc[row, col] = np.nan

## Hide upper triangle for cleaner visualisation
mask = np.triu(np.ones_like(corr_mat, dtype = bool))

## Heatmap
plt.figure(figsize = (16, 14))
sns.heatmap(corr_mat, annot = False, fmt = ".2f", mask = mask, cmap = 'coolwarm',
            center = 0, cbar_kws = {'label': 'Correlation coefficient'})

## Add *** for significant p-values (p < 0.05) in lower triangle
for i in range(len(corr_mat.columns)):
    for j in range(i):
        if p_val.iloc[i, j] < 0.05:
            plt.text(j+0.5, i+0.5, '***', ha = 'center', va = 'center',
                     color = 'black', fontsize = 12)

## Map visualisation
plt.title('Correlation Matrix with Significance at 5%')
plt.show()
```

<img width="558" alt="image" src="https://github.com/user-attachments/assets/61b89500-48ff-48e6-80a7-77041f37bdfe">


## 2.4. Time distribution

```
sns.histplot(creditcard['Time'], bins = 30, kde = True, color = 'blue', edgecolor = 'black', stat = 'count')
plt.xlabel('Time (in seconds)')
plt.ylabel('Number of transactions')
plt.title('Time Distribution')
plt.show()
```

<img width="584" alt="image" src="https://github.com/user-attachments/assets/cc7c13e5-5e68-49e1-9f74-01760e19338a">


## 2.5. Distribution of Time based on Class

```
sns.boxplot(x = 'Class', y = 'Time', data = creditcard, palette = ['blue', 'red'])
plt.title('Time Distribution based on Transaction Type')
plt.xlabel('Non-fraud (0) vs Fraud (1)')
plt.ylabel('Time (in seconds)')
plt.show()
```

<img width="584" alt="image" src="https://github.com/user-attachments/assets/29225dc2-0ba1-448f-9c2f-4b1b82ec86f0">


## 2.6. Amount distribution

```
sns.histplot(creditcard['Amount'], bins = 30, kde = True, color = 'blue', edgecolor = 'black', stat = 'count')
plt.xlabel('Transaction amount (in Euro)')
plt.ylabel('Number of transactions')
plt.title('Transaction Amount Distribution')
plt.show()
```

<img width="559" alt="image" src="https://github.com/user-attachments/assets/10c12927-5172-45e7-ad53-853fb2340419">


## 2.7. Distribution of Amount based on Class

```
## Create a 1x2 grid of subplots
fig, axes = plt.subplots(1, 2, figsize = (16, 6))

## Without outliers
sns.boxplot(ax = axes[0], x = 'Class', y = 'Amount', data = creditcard, palette = ['blue', 'red'], showfliers = False)
axes[0].set_title('Amount Distribution based on Transaction Type (Without Outliers)')
axes[0].set_xlabel('Non-fraud (0) vs Fraud (1)')
axes[0].set_ylabel('Transaction Amount (in Euro)')

## With outliers
sns.boxplot(ax = axes[1], x = 'Class', y = 'Amount', data = creditcard, palette = ['blue', 'red'])
axes[1].set_title('Amount Distribution based on Transaction Type (With Outliers)')
axes[1].set_xlabel('Non-fraud (0) vs Fraud (1)')
axes[1].set_ylabel('Transaction Amount (in Euro)')

## Show plots
plt.tight_layout()
plt.show()
```

<img width="1114" alt="image" src="https://github.com/user-attachments/assets/bedad3e5-ce3c-4fb9-875f-33b0c711d3f2">



# 3. Fraud detection models
## 3.1. Variables extraction

```
X = creditcard.drop('Class', axis = 1)
y = creditcard['Class']
```


## 3.2. Assumptions checking
To check for independence of observations, assumption for Random Forest, XGBoost, and Extra-Trees.

### a. Durbin-Watson test
Results of Durbin-Watson test:
* 0 to < 2: positive autocorrelation
* Around 2: no autocorrelation
* \> 2 to 4: negative autocorrelation

```
import statsmodels.api as sm                           # for OLS regression
from statsmodels.stats.stattools import durbin_watson  # for Durbin-Watson test
```

```
## Fit multiple linear regression model
x = sm.add_constant(X)      # add constant term to predictor
linear = sm.OLS(y, x).fit()

## Durbin-Watson test
dw_stat = durbin_watson(linear.resid)
print('Durbin-Watson statistic = %.2f' % dw_stat)
```

*Durbin-Watson statistic = 1.97*

Durbin-Watson statistic near 2 indicates little or no autocorrelation in the residuals. Therefore, the assumption of significance of observations is satisfied.


### b. Residuals over Time plot

```
plt.plot(linear.resid)
plt.title('Residuals over Time')
plt.show()
```

<img width="546" alt="image" src="https://github.com/user-attachments/assets/52fcf886-72cb-40d6-b5d9-18548cc90d3a">

The residuals distribute around zero line with no specific pattern, indicates no trend over time.


### c. Autocorrelation plot

```
pd.plotting.autocorrelation_plot(linear.resid)
plt.title('Autocorrelation Plot')
plt.show()
```

<img width="577" alt="image" src="https://github.com/user-attachments/assets/b25701b8-a3cd-4c37-9b34-26012ad7beec">

The plot shows no positive or negative autocorrelation for all lags.


## 3.3. Train-test split

```
from sklearn.model_selection import train_test_split  # for splitting data
```

```
## Train-test split with ratio 8:2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 112)
```


## 3.4. Data normalisation

```
from sklearn.preprocessing import StandardScaler  # for z-score standardisation
```

```
## Fit scaler on training data
scaler = StandardScaler()
scaler.fit(X_train)

## Standardise both training and testing sets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

## Transform data from array back to dataframe
X_train_scaled = pd.DataFrame(X_train_scaled, columns = X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns = X.columns)
```


## 3.5. Model 1: Tomek link removal + Random Forest
### a. Tomek link removal
#### Resample dataset

```
from imblearn.under_sampling import TomekLinks  # for Tomek link removal
```

```
## Create TomekLinks object
tomeklink = TomekLinks(sampling_strategy = 'majority')  # remove only majority class

## Resample the dataset
X_res, y_res = tomeklink.fit_resample(X_train_scaled, y_train)
```


#### Dataset visualisation - before and after removal

```
from collections import Counter        # for counting number of elements
from sklearn.decomposition import PCA  # for reducing data dimensions
```

```
## Original dataset visualisation
# perform PCA for visualization purpose
pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train_scaled)
# determine which data points to be removed
remove_index = np.setdiff1d(np.arange(len(X_train_scaled)), tomeklink.sample_indices_)
# create scatter plot for the original dataset
plt.subplot(1, 2, 1)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], color = 'blue', alpha = 0.5, label = 'Original Data')
plt.scatter(X_train_pca[remove_index, 0], X_train_pca[remove_index, 1], color = 'red', label = 'Points to be Removed')
plt.title('Original Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

## Resampled dataset visualisation
# perform PCA for visualization purpose
X_res_pca = pca.transform(X_res)
# create scatter plot for the resampled dataset
plt.subplot(1, 2, 2)
plt.scatter(X_res_pca[:, 0], X_res_pca[:, 1], color = 'blue', alpha = 0.5)
plt.title('Resampled Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

## Show plots
plt.tight_layout()
plt.show()

## Class distribution
print('Original dataset shape:', Counter(y_train))  # original dataset
print('Resampled dataset shape:', Counter(y_res))   # resampled dataset
```

<img width="622" alt="image" src="https://github.com/user-attachments/assets/efa327eb-9bf7-47b4-b482-744ac0908430">

*Original dataset shape: Counter({0: 227448, 1: 397})*

*Resampled dataset shape: Counter({0: 227426, 1: 397})*


### b. Random Forest
#### Cross validation to find optimal model

```
from sklearn.ensemble import RandomForestClassifier                # for Random Forest model
from sklearn.model_selection import StratifiedKFold, GridSearchCV  # for performing cross validation
from sklearn.metrics import make_scorer, average_precision_score   # for computing AUPRC score
```

```
## Define model
rand_forest = RandomForestClassifier(class_weight = 'balanced', random_state = 112)

## Define parameter grid for GridSearchCV
param_grid = {
    'max_depth': 2**np.array(range(1, 11), dtype = 'int'),
    'min_samples_leaf': [1, 3, 5]
}

## Ensure each fold has similar class distribution
cv = StratifiedKFold(n_splits = 3)

## Define custom scoring for AUPRC
auprc_scorer = make_scorer(average_precision_score, needs_proba = True)

## Create GridSearchCV object
grid_search = GridSearchCV(rand_forest, param_grid, cv = cv, scoring = auprc_scorer, n_jobs = -1)

## Perform grid search
grid_search.fit(X_res, y_res)

## Print best score and corresponding parameters
print('Best AUPRC score = %.2f' % grid_search.best_score_, 'achieved at the following parameters:')
print(grid_search.best_params_)
```

*Best AUPRC score = 0.84 achieved at the following parameters:*

*{'max_depth': 64, 'min_samples_leaf': 3}*


#### Model training

```
best_rand_forest = grid_search.best_estimator_  # model with best parameters
best_rand_forest.fit(X_res, y_res)
```


#### Model testing

```
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, average_precision_score  # confusion matrix & AUPRC score
```

```
## Prediction
y_pred = best_rand_forest.predict(X_test_scaled)

## Confusion matrix
con_mat = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = con_mat)
disp.plot(cmap = plt.cm.Blues)
plt.title('Confusion Matrix of Model 1')
plt.show()

## Categorical accuracy
fraud_acc_model1 = con_mat[1, 1] / np.sum(con_mat, axis = 1)[1] * 100
legit_acc_model1 = con_mat[0, 0] / np.sum(con_mat, axis = 1)[0] * 100
print('Fraudulent accuracy = %.2f%%' % fraud_acc_model1)
print('Genuine accuracy = %.2f%%' % legit_acc_model1)

## AUPRC score
y_pred_prob_model1 = best_rand_forest.predict_proba(X_test_scaled)[:, 1]  # probability for test set
auprc_model1 = average_precision_score(y_test, y_pred_prob_model1)
print('AUPRC score = %.2f' % auprc_model1)
```

<img width="516" alt="image" src="https://github.com/user-attachments/assets/7caf3012-e2e5-4bf4-947e-2c993edcfa23">

*Fraudulent accuracy = 76.84%*

*Genuine accuracy = 99.98%*

*AUPRC score = 0.85*


## 3.6. Model 2: K-means + XGBoost
### a. K-means
#### Elbow method to find a range of possible K

```
from sklearn.cluster import KMeans  # for K-means
```

```
## Range of K values
k_val = range(2, 101)

## Initialise list of inertias
inertias = []

## Compute inertia for different values of K
for k in k_val:
    kmeans = KMeans(n_clusters = k, n_init = 10, random_state = 112)
    kmeans.fit(X_train_scaled)
    inertias.append(kmeans.inertia_)

## Elbow graph
plt.plot(k_val, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal K')
plt.show()
```

<img width="556" alt="image" src="https://github.com/user-attachments/assets/dac865b2-7699-46fc-a6a3-c3b094dc7639">

Optimal range of K is between 20 and 40.


#### Silhouette score to find best K

```
from sklearn.metrics import silhouette_score  # for computing Silhouette score
```

```
## Range of K values
k_val = range(20, 41, 2)

## Initialise list of silhouette scores
sil_scores = []

## Compute silhouette scores for different values of K
for k in k_val:
    kmeans = KMeans(n_clusters = k, n_init = 10, random_state = 112)
    kmeans.fit(X_train_scaled)
    score = silhouette_score(X_train_scaled, kmeans.labels_)
    sil_scores.append(score)
    print('Silhouette score = %.2f' % score, 'at K =', k)

## Silhouette scores plot
plt.plot(k_val, sil_scores, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different Values of K')
plt.show()

## Best K value
best_k = k_val[np.argmax(sil_scores)]
print('Best K value =', best_k, 'achieved at the highest Silhouette score of %.2f' % max(sil_scores))
```

*Silhouette score = 0.08 at K = 20*

*Silhouette score = 0.08 at K = 22*

*Silhouette score = 0.08 at K = 24*

*Silhouette score = 0.07 at K = 26*

*Silhouette score = 0.07 at K = 28*

*Silhouette score = 0.09 at K = 30*

*Silhouette score = 0.08 at K = 32*

*Silhouette score = 0.08 at K = 34*

*Silhouette score = 0.08 at K = 36*

*Silhouette score = 0.08 at K = 38*

*Silhouette score = 0.08 at K = 40*

<img width="577" alt="image" src="https://github.com/user-attachments/assets/2684afdd-e24e-4049-9fc3-9fc543bf072f">

*Best K value = 30 achieved at the highest Silhouette score of 0.09*


### b. Clustering & XGBoost
#### Cross-validation to find optimal model

```
from sklearn.base import BaseEstimator, TransformerMixin  # for building K-means transformation function
from xgboost import XGBClassifier                         # for XGBoost
from sklearn.pipeline import Pipeline                     # for stacking multiple functions together
```

```
## Negative-to-positive class ratio
neg_pos_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)

## Define function to integrate K-means clustering
class KMeansTransformer(BaseEstimator, TransformerMixin):
    # define parameters
    def __init__(self, n_clusters, n_init, random_state):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state
    # define and fit K-means model to cluster
    def fit(self, X, y = None):
        self.kmeans = KMeans(n_clusters = self.n_clusters, n_init = self.n_init, random_state = self.random_state)
        self.kmeans.fit(X)
        return self
    # integrate clusters
    def transform(self, X, y = None):
        clusters = self.kmeans.predict(X)
        X_new = pd.DataFrame(X).copy()
        X_new['cluster'] = clusters
        return X_new

## Define pipeline
pipeline = Pipeline([
    ("kmeans", KMeansTransformer(n_clusters = best_k, n_init = 10, random_state = 112)),
    ("xgb", XGBClassifier(scale_pos_weight = neg_pos_ratio, random_state = 112))
])

## Define parameter grid for GridSearchCV
param_grid = {
    'xgb__max_depth': 2**np.array(range(1, 11), dtype = 'int'),
    'xgb__gamma': [0, 0.1],
    'xgb__reg_alpha': [0, 0.1],
    'xgb__reg_lambda': [0, 0.5]
}

## Ensure each fold has similar class distribution
cv = StratifiedKFold(n_splits = 3)

## Define custom scoring for AUPRC
auprc_scorer = make_scorer(average_precision_score, needs_proba = True)

## Create GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv = cv, scoring = auprc_scorer, n_jobs = -1)

## Perform grid search
grid_search.fit(X_train_scaled, y_train)

## Print best score and corresponding parameters
print('Best AUPRC score = %.2f' % grid_search.best_score_, 'achieved at the following parameters:')
print(grid_search.best_params_)
```

*Best AUPRC score = 0.85 achieved at the following parameters:*

*{'xgb__gamma': 0, 'xgb__max_depth': 8, 'xgb__reg_alpha': 0.1, 'xgb__reg_lambda': 0}*


#### Model training

```
best_pipeline = grid_search.best_estimator_  # model with best parameters
best_pipeline.fit(X_train_scaled, y_train)
```


#### Model testing

```
## Prediction
y_pred = best_pipeline.predict(X_test_scaled)

## Confusion matrix
con_mat = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = con_mat)
disp.plot(cmap = plt.cm.Blues)
plt.title('Confusion Matrix of Model 2')
plt.show()

## Categorical accuracy
fraud_acc_model2 = con_mat[1, 1] / np.sum(con_mat, axis = 1)[1] * 100
legit_acc_model2 = con_mat[0, 0] / np.sum(con_mat, axis = 1)[0] * 100
print('Fraudulent accuracy = %.2f%%' % fraud_acc_model2)
print('Genuine accuracy = %.2f%%' % legit_acc_model2)

## AUPRC score
y_pred_prob_model2 = best_pipeline.predict_proba(X_test_scaled)[:, 1]  # probability for test set
auprc_model2 = average_precision_score(y_test, y_pred_prob_model2)
print('AUPRC score = %.2f' % auprc_model2)
```

<img width="513" alt="image" src="https://github.com/user-attachments/assets/3c624bf3-9f67-473e-a035-d0d9c8dae56d">

*Fraudulent accuracy = 83.16%*

*Genuine accuracy = 99.99%*

*AUPRC score = 0.86*


## 3.7. Model 3: Extra-Trees + DNN
### a. Extra-Trees
#### Cross validation to find optimal model

```
from sklearn.ensemble import ExtraTreesClassifier  # for Extra-Trees
```

```
## Define model
extra_trees = ExtraTreesClassifier(class_weight = 'balanced', random_state = 112)

## Define parameter grid for GridSearchCV
param_grid = {
    'max_depth': 2**np.array(range(1, 11), dtype = 'int'),
    'min_samples_leaf': [1, 3, 5]
}

## Ensure each fold has similar class distribution
cv = StratifiedKFold(n_splits = 3)

## Define custom scoring for AUPRC
auprc_scorer = make_scorer(average_precision_score, needs_proba = True)

## Create GridSearchCV object
grid_search = GridSearchCV(extra_trees, param_grid, cv = cv, scoring = auprc_scorer, n_jobs = -1)

## Perform grid search
grid_search.fit(X_train_scaled, y_train)

## Print best score and corresponding parameters
print('Best AUPRC score = %.2f' % grid_search.best_score_, 'achieved at the following parameters:')
print(grid_search.best_params_)
```

*Best AUPRC score = 0.85 achieved at the following parameters:*

*{'max_depth': 64, 'min_samples_leaf': 1}*


#### Model training

```
best_extra_trees = grid_search.best_estimator_  # model with best parameters
best_extra_trees.fit(X_train_scaled, y_train)
```


#### Visualise feature importances

```
## Extract feature importances
feature_importances = best_extra_trees.feature_importances_

## Create a dataframe for visualisation
importances_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

## Sort dataframe by importance
importances_df = importances_df.sort_values(by = 'Importance', ascending = False)

## Plot feature importances
sns.barplot(x = 'Importance', y = 'Feature', data = importances_df, color = 'blue')
plt.title('Feature Importances from Extra-Trees Classifier')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
```

<img width="590" alt="image" src="https://github.com/user-attachments/assets/c5a8e7cf-4c0b-4857-8c15-ef34c7e13bae">


#### Select feature importances

```
## Select top 15 important features
importances_index = np.argsort(feature_importances)[-15:]
X_train_importances = X_train_scaled.iloc[:, importances_index]
X_test_importances = X_test_scaled.iloc[:, importances_index]
```


### b. DNN
#### Transform data into tensors

```
import tensorflow as tf  # for tensor creation & DNN construction
```

```
X_train_tensor = tf.constant(X_train_importances, dtype = tf.float32)
X_test_tensor = tf.constant(X_test_importances, dtype = tf.float32)
y_train_tensor = tf.constant(y_train.values, dtype = tf.float32)
y_test_tensor = tf.constant(y_test.values, dtype = tf.float32)
```


#### Define model

```
import random  # for setting seed
```

```
## Define custom loss function combining categorical loss and MAE
def custom_loss(y_true, y_pred):
    # calculate categorical loss
    categorical_loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    # calculate MAE loss
    mae_loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    # combine losses
    combined_loss = categorical_loss + mae_loss
    return combined_loss

## For reproducibility
def set_seed(seed = 112):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

## Define DNN model
def DNN(X_train, y_train):
    # set seed
    set_seed()
    # build structure
    model = tf.keras.Sequential([
        # layer 1
        tf.keras.layers.Dense(128, input_shape = (X_train.shape[1],), activation = 'relu'),
        # layer 2
        tf.keras.layers.Dense(64, activation = 'relu'),
        # layer 3
        tf.keras.layers.Dense(32, activation = 'relu'),
        # layer 4
        tf.keras.layers.Dense(16, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')  # transform output into values between 0 and 1
    ])
    # compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer = optimizer, loss = custom_loss, metrics = [tf.keras.metrics.AUC(name = 'auprc', curve = 'PR')])
    # early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)
    # learning rate scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 5, min_lr = 1e-6)
    # calculate class weights
    class_weight = {0: 1, 1: np.sum(y_train == 0) / np.sum(y_train == 1)}
    # fit model
    model.fit(X_train, y_train, epochs = 50, batch_size = 32, validation_split = 0.1,
              callbacks = [early_stopping, lr_scheduler], class_weight = class_weight)
    return model
```


#### Model training

```
## Train model
dnn = DNN(X_train_tensor, y_train_tensor)

## Check predictions range
unique_predictions = np.unique(dnn.predict(X_test_tensor))
print('Number of unique combined predictions:', len(unique_predictions))

## Check if predictions are not all the same
if len(unique_predictions) < 10:
    print('Warning: The model predictions might be collapsing to a single value.')
else:
    print('Model predictions have a reasonable variance.')
```

*Number of unique combined predictions: 56740*

*Model predictions have a reasonable variance.*


#### Model testing

```
## Prediction
y_pred_prob_model3 = dnn.predict(X_test_tensor).reshape(-1)  # ensure it's a row vector
y_pred = (y_pred_prob_model3 >= 0.5).astype(int)             # convert probability into binary classification

## Confusion matrix
con_mat = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = con_mat)
disp.plot(cmap = plt.cm.Blues)
plt.title('Confusion Matrix of Model 3')
plt.show()

## Categorical accuracy
fraud_acc_model3 = con_mat[1, 1] / np.sum(con_mat, axis = 1)[1] * 100
legit_acc_model3 = con_mat[0, 0] / np.sum(con_mat, axis = 1)[0] * 100
print('Fraudulent accuracy = %.2f%%' % fraud_acc_model3)
print('Genuine accuracy = %.2f%%' % legit_acc_model3)

## AUPRC score
auprc_model3 = average_precision_score(y_test, y_pred_prob_model3)
print('AUPRC score = %.2f' % auprc_model3)
```

<img width="514" alt="image" src="https://github.com/user-attachments/assets/132f4024-0982-4ba1-a154-9e199e755022">

*Fraudulent accuracy = 83.16%*

*Genuine accuracy = 99.95%*

*AUPRC score = 0.85*



# 4. Model comparison
## 4.1. Categorical accuracy

```
## Prepare plot's features
fraud_accuracies = [fraud_acc_model1, fraud_acc_model2, fraud_acc_model3]
legit_accuracies = [legit_acc_model1, legit_acc_model2, legit_acc_model3]
models = ['Model 1', 'Model 2', 'Model 3']
pos = np.arange(len(models))  # positions for the groups
width = 0.35                  # width of the bars

## Create supblots
fig, ax = plt.subplots()

## Plot the bars
bars1 = ax.bar(pos - width/2, fraud_accuracies, width, color = 'red', label = 'Fraudulent Accuracy')
bars2 = ax.bar(pos + width/2, legit_accuracies, width, color = 'blue', label = 'Legitimate Accuracy')

## Add labels, title, and custom x-axis tick labels
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy of Fraudulent and Legitimate Transactions by Model')
ax.set_xticks(pos)
ax.set_xticklabels(models)
ax.legend(loc = 'lower left')

## Add bar labels
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('%.2f' % height, xy = (bar.get_x() + bar.get_width() / 2, height),
                    xytext = (0, 3),  # 3 points vertical offset
                    textcoords = "offset points", ha = 'center', va = 'bottom')
add_labels(bars1)
add_labels(bars2)

## Display bar chart
plt.tight_layout()
plt.show()
```

<img width="619" alt="image" src="https://github.com/user-attachments/assets/ee240c01-c738-4af8-a56a-1084a61ce267">


## 4.2. Precision-recall curves

```
from sklearn.metrics import precision_recall_curve  # for plotting precision-recall curve
```

```
## Prepare plot's features
model_predictions = [y_pred_prob_model1, y_pred_prob_model2, y_pred_prob_model3]
model_labels = ['Model 1', 'Model 2', 'Model 3']
colors = ['blue', 'green', 'red']

## Plot precision-recall curves
for idx, (model_prob, label, color) in enumerate(zip(model_predictions, model_labels, colors)):
    precision, recall, _ = precision_recall_curve(y_test, model_prob)
    auprc = average_precision_score(y_test, model_prob)
    plt.plot(recall, precision, lw = 2, color = color, label = f'{label} (AUPRC = {auprc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.show()
```

<img width="556" alt="image" src="https://github.com/user-attachments/assets/5ed12a36-c591-431a-a087-162d412fafc4">
