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

284807 rows × 31 columns


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

Fraud-to-non-fraud ratio = 0.17%


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
sns.histplot(creditcard['Time'], bins = 30, kde = True,
             color = 'blue', edgecolor = 'black', stat = 'count')
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





















