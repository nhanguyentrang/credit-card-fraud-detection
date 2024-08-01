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
