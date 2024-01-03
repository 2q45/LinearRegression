import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

!wget -q --show-progress "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%201%20-%205/Session%202b%20-%20Logistic%20Regression/cancer.csv"

data = pd.read_csv('cancer.csv')
data['diagnosis'].replace({'M':1, 'B':0}, inplace = True)
data.to_csv('cancer.csv')
del data
data_path = 'cancer.csv'

dataframe = pd.read_csv(data_path)

dataframe = dataframe[['diagnosis', 'perimeter_mean', 'radius_mean', 'texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean', 'symmetry_mean']]
dataframe['diagnosis_cat'] = pd.Categorical(dataframe['diagnosis'].astype(str), categories=['0', '1'], ordered=False).rename_categories(['0 (benign)', '1 (malignant)'])

train_df, test_df = train_test_split(dataframe, test_size = 0.2, random_state = 1)


X = ['radius_mean']
y = 'diagnosis'

X_train = train_df[X]
X_test = test_df[X]
print('X_train, our input variables:')
print(X_train.head())
print()

y_train = train_df[y]
y_test = test_df[y]
print('y_train, our output variable:')
print(y_train.head())

logreg_model = linear_model.LogisticRegression()

logreg_model.fit(X_train,y_train)
y_pred = logreg_model.predict(X_test)


test_df['predicted'] = y_pred.squeeze()
sns.catplot(x = 'radius_mean', y = 'diagnosis_cat', hue = 'predicted', data=test_df, order=['1 (malignant)', '0 (benign)'])
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
