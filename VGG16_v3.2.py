from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import os
import tensorflow as tf
sns.set_style('whitegrid')
from glob import glob
# test GPU
# tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

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

# create a column to store the full path of images for later reading
# You should change the below path when running at your computer
my_glob = glob('C:/Users/Zijian/OneDrive - Cranfield University/Group project/NIH/images/images/*.png')
print('Number of Observations: ', len(my_glob))
full_img_paths = {os.path.basename(x): x for x in my_glob}
df_NIH['path'] = df_NIH['Image Index'].map(full_img_paths.get)



# keep five normal & single desease 
labels = ['Effusion', 'Atelectasis', 'Infiltration', 'Pneumonia', 'Nodule', 'No Finding']

df_NIH = df_NIH[df_NIH['Finding Labels'].isin(labels)]

# add 3800 pneumonia images from other datasets
glob2 = glob('C:/Users/Zijian/Cranfield University/Muhammad Hasan Ali, Hasan - GroupProject/Research/PNEUMONIA/*.jpeg')
df_extraPhe = pd.DataFrame(glob2, columns=['path'])
df_extraPhe['Finding Labels'] = 'Pneumonia'

# concat the NIH pneumonia and other pneumonia images together
xray_data = pd.concat([df_NIH, df_extraPhe])

num = []
for i in labels:
	temp = len(xray_data[xray_data['Finding Labels'].isin([i])])
	num.append(temp)

# draw the data distribution
df_draw = pd.DataFrame(data={'labels':labels, 'num':num})
df_draw = df_draw.sort_values(by='num', ascending=False)
ax = sns.barplot(x='num', y='labels', data=df_draw, color="green")
# fig = ax.get_figure()
# fig.savefig('./fig/a.png')

# split data into train, test, validation
train_set, valid_set = train_test_split(xray_data, test_size = 0.02, random_state = 42)

train_set, test_set = train_test_split(train_set, test_size = 0.2, random_state = 8545)

# quick check to see that the training and test set were split properly
print("train set:", len(train_set))
print("test set:", len(test_set))
print("validation set:", len(valid_set))
print('full data set: ', len(xray_data))


# Create ImageDataGenerator, to perform significant image augmentation
# Utilizing most of the parameter options to make the image data even more robust
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

image_size = (128, 128) # image re-sizing target

train_generator = train_datagen.flow_from_dataframe(
	dataframe=train_set,
	directory=None,
	x_col='path', 
	y_col = 'Finding Labels',
	target_size=image_size,
	color_mode='rgb',
	batch_size=32,
	class_mode='categorical'
)

test_generator = train_datagen.flow_from_dataframe(
	dataframe=test_set,
	directory=None,
	x_col='path', 
	y_col = 'Finding Labels',
	target_size=image_size,
	color_mode='rgb',
	batch_size=64,
	class_mode='categorical'	
)


valid_X, valid_Y = next(test_datagen.flow_from_dataframe(
	dataframe=valid_set,
	directory=None,
	x_col='path', 
	y_col = 'Finding Labels',
	target_size=image_size,
	color_mode='rgb',
	batch_size= len(valid_set),
	class_mode='categorical'
))


# load the VGG16 network, ensuring the head FC layer sets are left off
baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(128, 128, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(6, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.5e-3), 
	loss='categorical_crossentropy', 
	metrics=['accuracy'])
model.summary()

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, 'VGG16_train')


# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer]



print("[INFO] training head...")
EPOCHS = 30
H = model.fit_generator(
        train_generator,
        steps_per_epoch=20,
        epochs=EPOCHS,
        validation_data=test_generator,
        validation_steps=20)



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
plt.savefig('performance2.png', ppi=300)

#@todo according to the performance curves choose the best parameters

# Using validation datasets to predict
print("[INFO] evaluating network...")
predval = model.predict(valid_X)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predval2 = np.argmax(predval, axis=1)

name = []
for i in train_generator.class_indices.keys():
	name.append(i)
# print(train_generator.class_indices) # name in dict
print(classification_report(valid_Y.argmax(axis=1), predval2, target_names=name))


scores = model.evaluate(valid_X, valid_Y, verbose=1)
print('Validation loss:', scores[0])
print('Validation accuracy:', scores[1])


#plot AUC
from sklearn.metrics import roc_curve, auc

fig, c_ax = plt.subplots(1, 1,figsize=(8,8))
for (i, label) in enumerate(train_generator.class_indices):
	fpr, tpr, thresholds = roc_curve(valid_Y[:, i].astype(int), predval[:, i])
	c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (label, auc(fpr, tpr)))

# Set labels for plot
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('./fig/auc_vgg16.png')


# print("[INFO] saving COVID-19 detector model...")
# model.save("model", save_format="h5")