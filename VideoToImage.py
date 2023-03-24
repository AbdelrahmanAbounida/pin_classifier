import cv2
import os
import glob
import shutil


class VideoToImage:

    def __init__(self,target_dataset_folder):

        self.base_dir = os.path.join(os. getcwd(), target_dataset_folder) 
        self.list_of_videos, self.class_names = self.get_list_videos()

        self.total_train = 0
        self.total_val = 0

        self.make_videos_dirs()  
    

    def get_list_videos(self):
        """ generate list of videos in the current directory to generate the dataset with labels same as the video names """

        list_of_videos =  glob.glob("./*.mp4")
        class_names = list(map(lambda x: x.split('./')[1].split('.')[0], list_of_videos))
        return list_of_videos, class_names



    def make_videos_dirs(self):
        """ create directories for classes """

        try:
            if not os.path.exists(f'{self.base_dir}'):
                os.makedirs(f'{self.base_dir}')
        except OSError:
                print ('Error: Creating directory of data')

        for class_name in self.class_names:
            try:
                if not os.path.exists(f'{self.base_dir}/{class_name}'):
                    os.makedirs(f'{self.base_dir}/{class_name}')
            except OSError:
                print ('Error: Creating directory of data') 
                
    

    def check_images_generation(self):
        """  return True if images have already been generated """

        if not self.list_of_videos:
            print("There is no videos in the given directory")
            return True

        for class_name in self.class_names:
            
            if glob.glob(f"{self.base_dir}/{class_name}/*.jpg") :
                print("Images have already been generated")
                return True

            if glob.glob(f"{self.base_dir}/train/{class_name}/*.jpg"):
                print("Images have been generated and moved to train or val directory")
                return True
        
        return False



    def generate_images(self):
        """ generate list of images from each existing video """

        if self.check_images_generation():
            return

        for index,video in enumerate(self.list_of_videos):
            cam = cv2.VideoCapture(video)
            currentframe = 0
            while(True):
                # reading from frame
                ret,frame = cam.read()
                if ret:
                    name = f'{self.base_dir}/{self.class_names[index]}/{self.class_names[index]}_' + str(currentframe) + '.jpg'
                    print ('Creating...' + name)
                    cv2.imwrite(name, frame)
                    currentframe += 1
                else:
                    break

            cam.release()
            cv2.destroyAllWindows()


    
    def create_train_val_dataset(self):
        """ create folders for train and validation dataset and move images to them  """

        if self.check_images_generation():
            return

        total_train = 0 # total number of training images
        total_val = 0 # total number of validation images

        for cl in self.class_names:
            img_path = os.path.join(self.base_dir, cl)
            images = glob.glob(img_path + '/*.jpg')
            print("{}: {} Images".format(cl, len(images)))
            train, val = images[:round(len(images)*0.8)], images[round(len(images)*0.8):]
            total_train += len(train)
            total_val+=len(val)
            
            for t in train:
                if not os.path.exists(os.path.join(self.base_dir, 'train', cl)):
                    os.makedirs(os.path.join(self.base_dir, 'train', cl))
                shutil.move(t, os.path.join(self.base_dir, 'train', cl))

            for v in val:
                if not os.path.exists(os.path.join(self.base_dir, 'val', cl)):
                    os.makedirs(os.path.join(self.base_dir, 'val', cl))
                shutil.move(v, os.path.join(self.base_dir, 'val', cl))
        
        self.total_train = total_train
        self.total_val = total_val

        return total_train, total_val 
