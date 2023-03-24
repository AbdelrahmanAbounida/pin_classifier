from tensorflow.keras.preprocessing.image import ImageDataGenerator
from VideoToImage import VideoToImage
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import PIL.Image as Image
import tensorflow as tf
import numpy as np
import glob
import os


class PinClassifier:

    def __init__(self,base_dir):
        self.base_dir = base_dir

        self.train_data_gen = None # train data generator : includes training images dataset
        self.val_data_gen = None # validation data generator : includes validation images dataset

        self.model = None # current Model
        self.history = None # model history

    def load_dataset(self,class_names=[],IMG_SHAPE = (384,384),BATCH_SIZE=32,horizontal_flip=True,zoom_range=0.5,shuffle=True,color_mode="rgb",class_mode="sparse"):

        train_dir = os.path.join(self.base_dir, 'train')
        val_dir = os.path.join(self.base_dir, 'val')
        IMG_SHAPE = (384,384)
        BATCH_SIZE = 32

        if not len(class_names):
            print("Please specify the list of classname")
            return
        
        for class_name in class_names:
            if not len(glob.glob(f"{train_dir}/{class_name}/*.jpg")):
                print("There is no images in train directory, double check generating images to that folder")
                return 
            if not len(glob.glob(f"{val_dir}/{class_name}/*.jpg")):
                print("There is no images in validation directory, double check generating images to that folder")
                return 

        image_train_gen = ImageDataGenerator(rescale=1.0/255,horizontal_flip=True,
                                            #zoom_range=0.5,rotation_range=45,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2
                                            )

        train_data_gen = image_train_gen.flow_from_directory(batch_size=BATCH_SIZE,directory=train_dir,target_size=IMG_SHAPE,shuffle=True,
                                                            class_mode=class_mode, # class_mode is int >> sparse_categorical_crossentropy
                                                            color_mode = color_mode)

        image_val_gen = ImageDataGenerator(rescale=1.0/255,horizontal_flip=True,
                                            #zoom_range=0.5,rotation_range=45,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2
                                            )

        val_data_gen = image_val_gen.flow_from_directory(batch_size=BATCH_SIZE,directory=train_dir,target_size=IMG_SHAPE,shuffle=True,
                                                            class_mode=class_mode, # class_mode is int >> sparse_categorical_crossentropy
                                                            color_mode = color_mode)

        self.train_data_gen = train_data_gen
        self.val_data_gen = val_data_gen

        return train_data_gen, val_data_gen
    

    def plot_samples(self,n=5):
        """ n: number of samples """

        if not self.train_data_gen:
            print("Please load training dataset first, ex: train_data_data_gen")
            return

        augmented_images = [self.train_data_gen[0][0][0] for i in range(n)]

        if not len(augmented_images):
            print("There is no images in training director... Please check loading them first")
        fig, axes = plt.subplots(1, 5, figsize=(20,20))
        axes = axes.flatten()

        for img, ax in zip( augmented_images, axes):
            ax.imshow(img)
        plt.tight_layout()
        plt.show()
    

    def num_train_val_dataset(self,class_names):
        total_train = 0
        total_val = 0
        for i in class_names:
            total_train+=len(os.listdir(os.path.join(self.base_dir,'train',i)))
            total_val+=len(os.listdir(os.path.join(self.base_dir,'val',i))) 
        
        return total_train, total_val
    

    def create_model(self,IMG_SHAPE):
        # custom model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SHAPE[0], IMG_SHAPE[1], 3)),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),

            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),

            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),

            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(2) # we have 2 categories now
        ])

        self.model = model

        return model
    
    def compile_model(self,model,optimizer='adam',metrics=['accuracy'],label_mode="int"):
        """ label_mode : int >> sparse_categorical_crossentropy loss, 
            label_mode : categorical >> categorical_crossentropy,
            label_mode : binary >> binary_crossentropy
        """
        loss = None
        if label_mode == "int":
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # produces a category index of the most likely matching category.

        elif label_mode == "categorical":
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True) # produces a one-hot array containing the probable match for each category

        elif label_mode == "binary":
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        else:
            raise TypeError("Given label mode is wrong. Please check it again")

        model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])
        
        return model.summary()
    

    def train_model(self,model,train_data_gen,val_data_gen,total_train,total_val,BATCH_SIZE,epochs):
        history = model.fit_generator(train_data_gen,
                                steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
                                epochs=epochs,
                                validation_data=val_data_gen,
                                validation_steps=int(np.ceil(total_val / float(BATCH_SIZE))))
        
        self.history = history
        return history
    

    def plot_acc_loss_estimates(self,EPOCHS):
        """ plot the relationship between increasing the number of epochs and the effect on model loss and accuracy """
        EPOCHS = EPOCHS
        # Visualizing training results
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(EPOCHS)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig('./foo.png')
        plt.show()
    

    def plot_predictions(self,class_names):

        image_batch, _ = next(self.val_data_gen)

        label_batch = class_names

        result_batch = model.predict(image_batch)

        plt.figure(figsize=(10,9))
        for n in range(30):
            plt.subplot(6,5,n+1)
            # plt.subplots_adjust(hspace = 0.3)
            plt.imshow(a[n])
            plt.title(label_batch[np.argmax(np.array(result_batch[n]))])
            plt.axis('off')
        _ = plt.suptitle("Model predictions ")
    

    def save_model(self,model,model_name):
        model.save(f'{model_name}.h5')
    
    def laod_model(self,model_name):
        return tf.keras.models.load_model(f'{model_name}.h5')

    
    
