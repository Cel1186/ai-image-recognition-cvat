# ai-image-recognition-cvat
Developed and trained a computer vision model to identify and classify cats and dogs.

Upload dataset.zip and unzip to /content

!mkdir -p /content/dataset/
!unzip /content/dataset.zip -d /content/dataset/

Upload the exported YOLO1.1 file along with their images for training and unzip to `/content`

!mkdir -p /content/dataset_annotated/
!unzip /content/dataset_annotated.zip -d /content/dataset_annotated/

Install `ultralytics` module

!pip install ultralytics

Split the files into two: 80% must be the train and 20% must be the validation for each label

# 1. Create the target folders
!mkdir -p /content/dataset_annotated/obj_train_data/dataset_cat_dog/
!mkdir -p /content/dataset_annotated/obj_val_data/dataset_cat_dog/

# 2. Move your CVAT text labels into the training folder
!mv -v /content/dataset_annotated/obj_train_data/*.txt /content/dataset_annotated/obj_train_data/dataset_cat_dog/

# 3. Move your raw images from dataset.zip into the exact same folder
!mv -v /content/dataset/* /content/dataset_annotated/obj_train_data/dataset_cat_dog/

# 4. Safely split 20 images and their matching text labels into the validation folder
!mv -v /content/dataset_annotated/obj_train_data/dataset_cat_dog/1.* /content/dataset_annotated/obj_val_data/dataset_cat_dog/
!mv -v /content/dataset_annotated/obj_train_data/dataset_cat_dog/2.* /content/dataset_annotated/obj_val_data/dataset_cat_dog/
# ... repeat for your validation range, or run the rest of Cell 4 safely!

Save the following lines as `dataset.yaml`

%%writefile dataset.yaml
train: /content/dataset_annotated/obj_train_data/dataset_cat_dog
val: /content/dataset_annotated/obj_val_data/dataset_cat_dog

nc: 2
names: ['cat', 'dog']


Train the model

from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # small pre-trained model

model.train(
    data='dataset.yaml',
    epochs=50,
    imgsz=640
)

Use your trained model to predict on sample images with no annotation
model = YOLO("/content/runs/detect/train/weights/best.pt")

Upload [dataset_predict.zip](https://drive.google.com/file/d/1e94wMLNHv5Jq9UYDFH3Gd3sLISQCBNfk/view?usp=drive_link) for prediction and unzip to `/content/pictures`
!mkdir -p /content/pictures
!unzip /content/dataset_predict.zip -d /content/pictures


results = model.predict("/content/pictures/dataset_cat_dog_predict/*.jpg", save=True, conf=0.15, iou=0.45)


from IPython.display import Image, display
from glob import glob

Display the predictions made by the model
for pic in glob('/content/runs/detect/predict/*.jpg'):
  display(Image(f"{pic}"))
