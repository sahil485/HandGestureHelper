import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback

##To import, organize, and preprocess images and then train model
#gesture_v4 model latest with high accuracy for thumb up and down

class Callback(Callback): #callback class from predefined keras superclass: stops training once accuracy is 95%
    def on_epoch_end(self, epochs, logs={}):
        if(logs.get('auc') > 0.95):
            self.model.stop_training = True

def network(trainingDat, valDat, trainingLab = None, testDat = None, testLab = None):
    CallbackTraining = Callback()

    network = Sequential([ #cr  eates a sequential CNN with various conv, pooling, dropout, and dense layers
        Conv2D(16, (3,3), activation = tf.nn.relu, input_shape= (40,30, 1)),
        Dropout(0.3),
        Conv2D(32, (3,3), activation = tf.nn.relu),
        MaxPooling2D(2,2),
        Dropout(0.3),
        Conv2D(32, (3, 3), activation=tf.nn.relu),
        Dropout(0.3),
        Conv2D(64, (3,3), activation = tf.nn.relu),
        Dropout(0.3),
        Conv2D(128,(3,3), activation = tf.nn.relu),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(100, activation = tf.nn.relu),
        Dense(4, activation = tf.nn.softmax) #softmax activation due to the desired action of multiclass classification
    ])
    AUC = tf.keras.metrics.AUC(
    num_thresholds=200, curve='ROC',
    summation_method='interpolation', name=None, dtype=None,
    thresholds=None, multi_label=False, num_labels=None, label_weights=None,
    from_logits=False
)
    network.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss = 'categorical_crossentropy',metrics = AUC) #compiles and starts learning of the network
    network.fit(trainingDat, epochs = 30, callbacks = [CallbackTraining], validation_data=valDat)
    return network

train_data_dir = r"images"

train_datagen = ImageDataGenerator(
    width_shift_range=6,
    rotation_range= 30,
    fill_mode= 'nearest',
    zoom_range=0.4,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(40,30),
    batch_size= 32,
    color_mode= 'grayscale',
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(40,30),
    batch_size= 32,
    color_mode= 'grayscale',
    class_mode='categorical',
    subset='validation') # set as validation data


Network = network(train_generator, validation_generator)
Network.save("gestures_modelv6")

