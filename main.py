from VideoToImage import VideoToImage
from PinClassifier import PinClassifier


def main(model_name="pin_model.h5",pin_dataset_folder="pin_dataset",IMG_SHAPE = (384,384),BATCH_SIZE = 32,EPOCHS = 2,load_model=False):
    """ generate_images: if True means there is no images and we need to convert videos to images, else >> images already_exist 
        pin_dataset: pin-images main floder name
    """

    if load_model:
        return tf.keras.models.load_model(model_name)

    # 1- generate images from videos
    print()
    print("Start generating images from videos....")

    video_to_images = VideoToImage(pin_dataset_folder)
    video_to_images.generate_images() # generate_images
    video_to_images.create_train_val_dataset() # create train and validation dataset

    
    # 2- initialize the required parameters
    IMG_SHAPE = IMG_SHAPE
    BATCH_SIZE = BATCH_SIZE
    EPOCHS = EPOCHS
    class_names = video_to_images.class_names

    # 3- load dataset from folders
    pin_classifier = PinClassifier(video_to_images.base_dir)

    try:
        train_data_gen, val_data_gen = pin_classifier.load_dataset(class_names=class_names,IMG_SHAPE=IMG_SHAPE,BATCH_SIZE=BATCH_SIZE)
    except:
        print("Failed loading train,val dataset")

    # 4- Show training dataset
    # pin_classifier.plot_samples(5)
                                                               
    # 5- Create Model
    model = pin_classifier.create_model(IMG_SHAPE)

    # 6- Compile Model
    pin_classifier.compile_model(model)

    # 7- Train Model
    print()
    print("Training Model....")
    total_train,total_val = pin_classifier.num_train_val_dataset(class_names)
    history = pin_classifier.train_model(model,train_data_gen,val_data_gen, total_train, total_val, BATCH_SIZE, EPOCHS)

    # 8- Save Model
    model.save(f'{model_name}')

    # 8- plot_acc_loss_estimates
    pin_classifier.plot_acc_loss_estimates(EPOCHS)

    # 9- plot predictions
    pin_classifier.plot_predictions(class_names)


if __name__ == "__main__":
    main()

    # if model already trained
    # model = main(model_name="pin_model.h5",load_model=True)

