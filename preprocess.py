import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
sns.set_style('whitegrid')

"""
We have two datasets, NIH and our own COVID-19.
NIH dataset contains more than 100,000 images of different thoracic disease.

During data preprocessing, we could preocess the two datasets seperately
"""

## Preprocess NIH dataset

# read NIH data, the csv
df_NIH = pd.read_csv(r'C:\Users\Zijian\OneDrive - Cranfield University\Group project\NIH\Data_Entry_2017.csv')
# only keep the illnesss label and image names
df_NIH = df_NIH[['Image Index', 'Finding Labels']]

# create new columns for each illness
# total 14 thoracic disease
pathology_list = ['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule','Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia', 'No Finding']

for pathology in pathology_list :
    df_NIH[pathology] = df_NIH['Finding Labels'].apply(lambda x: 1.0 if pathology in x else 0.0)

# draw out the disease distribution
plt.figure(figsize=(15,10))
fig, ax = plt.subplots()
data1 = pd.melt(df_NIH,
             value_vars = list(pathology_list),
             var_name = 'Category',
             value_name = 'Count')
data1 = data1.loc[data1.Count>0]
g=sns.countplot(y='Category', data=data1, ax=ax, order = data1['Category'].value_counts().index)
ax.set( ylabel="",xlabel="")

### This is for testing the code and choose 500 images randomly
df_NIH_ill = df_NIH[~df_NIH['Finding Labels'].isin(['No Finding'])]
df_NIH_heal= df_NIH[df_NIH['Finding Labels'].isin(['No Finding'])]

df_NIH_ill = df_NIH_ill.sample(n=500)
df_NIH_heal =df_NIH_heal.sample(n=100)

df_NIH_ill = df_NIH_ill.drop(['Finding Labels'], axis=1)
df_NIH_heal = df_NIH_heal.drop(['Finding Labels'], axis=1)

# add the health and ill together
xray = pd.concat([df_NIH_heal, df_NIH_ill])

# for draw out the training percent
total = xray[pathology_list].sum().sort_values(ascending= False) 
clean_labels_df = total.to_frame() # convert to dataframe for plotting purposes
sns.barplot(x = clean_labels_df.index[::], y= 0, data = clean_labels_df[::], color = "green"), plt.xticks(rotation = 90) # visualize results graphically

# create vector as ground-truth, will use as actuals to compare against our predictions later
xray['target_vector'] = xray.apply(lambda target: [target[pathology_list].values], 1).map(lambda target: target[0])







