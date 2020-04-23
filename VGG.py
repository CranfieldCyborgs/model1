# import matplotlib.gridspec as gridspec
# import matplotlib.ticker as ticker
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import cv2
import os
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
pathology_list = ['Cardiomegaly','Emphysema','Effusion','Nodule','Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia', 'No Finding'] #'Hernia',

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

# create vector as ground-truth, will use as actuals to compare against our predictions later
# rename "xray" to classify other images
xray = pd.concat([df_NIH_heal, df_NIH_ill])

# for draw out the training percent
total = xray[pathology_list].sum().sort_values(ascending= False) 
clean_labels_df = total.to_frame() # convert to dataframe for plotting purposes
sns.barplot(x = clean_labels_df.index[::], y= 0, data = clean_labels_df[::], color = "green"), plt.xticks(rotation = 90) # visualize results graphically


# read images 
print("[INFO] loading images...")
data = []
imagename = xray['Image Index'].values.tolist()
prepath= 'C:\\Users\\Zijian\\OneDrive - Cranfield University\\Group project\\NIH\\images\\images\\'

for each in imagename:
    image = cv2.imread(prepath+each)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))
    data.append(image)

# all values are in the range from 0 to 1
data = np.array(data)/255.0

# create labels from Xray dataframe
xray['labels'] = xray.apply(lambda target: [target[pathology_list].values], 1).map(lambda target: target[0])
labels = np.stack(xray['labels'].values)

labels = labels.astype(float)



# the data for training and the remaining 33% for testing
print("[INFO] create train test datasets...")
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.33, random_state=42) #stratify=labels,


## model part 

# # some basic setting
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
# ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
# ap.add_argument("-m", "--model", type=str, default="covid19.model", help="path to output loss/accuracy plot")
# args = vars(ap.parse_args())

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 25
BS = 8

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=15,
	fill_mode="nearest")


# load the VGG16 network, ensuring the head FC layer sets are left off
baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(14, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="mean_squared_error", optimizer=opt,
	metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit_generator(
	trainAug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)


# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs))


# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity

cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
# show the confusion matrix, accuracy, sensitivity, and specificity

print(cm)
print("acc: {:.6f}".format(acc))
print("sensitivity: {:.6f}".format(sensitivity))
print("specificity: {:.6f}".format(specificity))


# plot the training loss and accuracy

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

plt.savefig('performance1.png', ppi=300)
# serialize the model to disk

print("[INFO] saving COVID-19 detector model...")
model.save("model", save_format="h5")