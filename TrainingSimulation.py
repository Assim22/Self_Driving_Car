
print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utlis import *
from sklearn.model_selection import train_test_split

#### STEP 1
path = 'myData'
data = importDataInfo(path)

#### Step 2
data = balanceData(data,display=False)

#### step 3

imagesPath,steering = loadData(path,data)

#print(imagesPath[0],steering[0])

#### STEP 4
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath,steering,test_size = 0.2, random_state=5)
print('Total Training Images',len(xTrain))
print('Total Validation Images',len(xVal))

#### step 5
model = createModel()
model.summary()
#### step 6

#### step 7

#### step 8


#### step 9
history = model.fit(batchGen(xTrain,yTrain,100,1),steps_per_epoch=400,epochs=12,
          validation_data=batchGen(xVal,yVal,100,0),validation_steps=200)

#### step 10
model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validaion'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()




