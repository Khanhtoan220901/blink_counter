import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import datetime
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import Model, load_model
###############################

path = "./dataset/OACE"
testRatio = 0.5
imageDimensions = (32,32,3)

batchSizeValue = 50

epochsValue = 10
stepsPerEpochValue = 2000
###############################
images = []
classNo = []
myList = os.listdir(path)
print("Total No of Classes Detected",len(myList))
noOfClasses = len(myList)
print("Importing Classes ......")

for x in myList:
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg,(imageDimensions[0],imageDimensions[1]))
        images.append(curImg)
        #classNo.append(x)
        if x == 'open':
            classNo.append(1)
        elif x == 'close':
            classNo.append(0)
    print(x,end= " ")
print(" ")

images = np.array(images)
classNo = np.array(classNo)

print(images.shape)

##### Spliting the data
X_train, X_test, Y_train, Y_test = train_test_split(images,classNo, test_size=testRatio)
#X_train, X_validation, Y_train, Y_validation = train_test_split(X_train,Y_train, test_size=valRatio)


print(X_train.shape)
print(X_test.shape)
#print(X_validation.shape)

numOfSamples = []
for x in range(0,noOfClasses):
    #print(len(np.where(Y_train==x)[0]))
    numOfSamples.append(len(np.where(Y_train==x)[0]))
print(numOfSamples)

plt.figure(figsize=(10,5))
plt.bar(range(0,noOfClasses),numOfSamples)
plt.title("No of Images for each Class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img = cv2.equalizeHist(img)
    #img = img/255
    return img

#img = preProcessing(X_train[32])
#img = cv2.resize(img,(300,300))
#cv2.imshow("PreProcessed",img)
#cv2.waitKey(0)


X_train = np.array(list(map(preProcessing,X_train)))
X_test = np.array(list(map(preProcessing,X_test)))
#X_validation = np.array(list(map(preProcessing,X_validation)))


X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
#X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)

np.save('./dataset/x_train.npy', X_train)
np.save('./dataset/y_train.npy', Y_train)
np.save('./dataset/x_test.npy', X_test)
np.save('./dataset/y_test.npy', Y_test)


X_train = np.load('./dataset/x_train.npy').astype(np.float32)
Y_train= np.load('./dataset/y_train.npy').astype(np.float32)
X_test= np.load('./dataset/x_test.npy').astype(np.float32)
Y_test= np.load('./dataset/y_test.npy').astype(np.float32)



train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(
    x=X_train, y=Y_train,
    batch_size=32,
    shuffle=True
)

val_generator = val_datagen.flow(
    x=X_test, y=Y_test,
    batch_size=32,
    shuffle=False
)


#Y_train = to_categorical(Y_train,noOfClasses)
#Y_test = to_categorical(Y_test,noOfClasses)
#Y_validation = to_categorical(Y_validation,noOfClasses)

def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNode = 500

    model = Sequential()
    model.add(Input(shape=(32, 32, 1)))
    model.add(((Conv2D(noOfFilters,sizeOfFilter1,input_shape=(imageDimensions[0],
                                                              imageDimensions[1],
                                                              1),activation='relu'
                                                              ))))

    model.add(((Conv2D(noOfFilters, sizeOfFilter1, activation='relu'))))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu'))))
    model.add(((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu'))))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = myModel()
print(model.summary())


start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
history = model.fit_generator(train_generator,
                              epochs=100,
                              validation_data=val_generator,
                              callbacks=[
                                  ModelCheckpoint('models/%s.h5' % (start_time), monitor='val_accuracy', save_best_only=True,
                                                  mode='max', verbose=1),
                                  ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, verbose=1, mode='auto', min_lr=1e-06)
                              ]
                              )

model = load_model('models/2022_06_04_15_14_07.h5')
y_pred = model.predict(X_test/255.)
y_pred_logical = (y_pred > 0.5).astype(np.int)

cm = confusion_matrix(Y_test, y_pred_logical)
sns.heatmap(cm, annot=True)

TP = cm[0][0]
TN = cm[1][1]
FP = cm[0][1]
FN = cm[1][0]
Accuracy = (TP + TN)/len(y_pred)
Precision = TP/(TP + FP)
Recall = TP/(TP + FN)
F1 = 2*((Recall * Precision)/(Recall + Precision))
print("Accuracy: ", Accuracy)
print("Precision: ", Precision)
print("Recall: ", Recall)
print("F-Score: ", F1)
print("Corrected: ", (TP + TN))
print("Falsed: ", (FP + FN))
print("")

ax = sns.distplot(y_pred, kde=False)

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_logical)
plt.figure(3)
plt.plot(fpr, tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.savefig("3.png")

plt.figure(1)
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.savefig("1.png")

plt.figure(2)
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.savefig("2.png")

plt.show()




