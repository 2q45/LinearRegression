#Boundaries when seperating data

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import os
#@title Run this to download your data! { display-mode: "form" }
# Load the data!
import pandas as pd
from sklearn import metrics

!wget -q --show-progress "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%201%20-%205/Session%202b%20-%20Logistic%20Regression/cancer.csv"

data = pd.read_csv('cancer.csv')
data['diagnosis'].replace({'M':1, 'B':0}, inplace = True)
data.to_csv('cancer.csv')
del data
data_path = 'cancer.csv'

# Read in the data from the CSV file
dataframe = pd.read_csv(data_path)

# Keep only the desired columns and create a new categorical column
dataframe = dataframe[['diagnosis', 'perimeter_mean', 'radius_mean', 'texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean', 'symmetry_mean']]
dataframe['diagnosis_cat'] = pd.Categorical(dataframe['diagnosis'].astype(str), categories=['0', '1'], ordered=False).rename_categories(['0 (benign)', '1 (malignant)'])

# Use the boundary variable as needed in subsequent code

def boundary_classifier(target_boundary, radius_mean_series):
  result = [] #fill this in with predictions!
  for i in radius_mean_series:
    if i > target_boundary:
      result.append(1)
    else:
      result.append(0)

  return result

chosen_boundary = 15 #Try changing this!

y_pred = boundary_classifier(chosen_boundary, dataframe['radius_mean'])
dataframe['predicted'] = y_pred

y_true = dataframe['diagnosis']

sns.catplot(x = 'radius_mean', y = 'diagnosis_cat', hue = 'predicted', data = dataframe, order=['1 (malignant)', '0 (benign)'])
plt.plot([chosen_boundary, chosen_boundary], [-.2, 1.2], 'g', linewidth = 2)
