# Tensorflow-Object-Dedection-API-Pretrained-Faster-RCNN-Model
Setups

This project was realized by creating a ”Virtual Environment” over ”Anaconda Prompt”. First of all, Anaconda must be installed on the computer.
In this study, Tensorflow-GPU v1.15 was used. 
In order to use Tensorflow GPU, first of all, it is necessary to download
CUDA and cuDNN suitable for the video card and suitable
for the Tensorflow version used.
1. First, a file named “tensorflow2” was opened in the C:/ directory and the
"https://github.com/tensorflow/models" repository was cloned here.(TensorFlow Object Detection
API repository)
2. Let’s download the trained model named ”faster rcnn inception v2 coco” from
the pre-trained faster rcnn models from "http://download.tensorflow.org/models/object detect. This file has been extracted to the
”C:/tensorflow2/models/research/object detection” folder.
3. The folder named “object detection” in the relevant Github repo https://github.com/ipeksonyildirim/Tensorflow-Objecwas cloned and extracted into the ”../object detection”
folder.
4. After doing these operations, let’s create a "Virtual Environment" over Anaconda

CREATING ANACONDA ENVIRONMENT AND REQUIREMENTS

conda create -n myenv python=3.6 
conda install tensorflowgpu==1.15.0 
conda install -c anaconda protobuf

After cloning the repo (link), install it from the requirements.txt file: 
pip install -r requirements.txt,
When the "Virtual Environment" to be used is activated, 
the Pythonpath needs to be set, and the following command should always be run whenever it is activated.
set PYTHONPATH=C:\tensorflow2\models;C:\tensorflow2\models\research;C:\tensorflow2\models\research\slim

Change directories to the ”C:/tensorflow2/model/research”
directory in the Prompt Then run the following command
cd C:\tensorflow2\models\research
protoc object_detection/protos/*.proto --python_out=.
Lastly run setup.py:
python setup.py build
python setup.py install

Preparing Dataset

In this project using prepared dataset PASCALVOC2012. This dataset is downloaded through
http://host.robots.ox.ac.uk/pascal/VOC/.
Then, test and training data should be moved to "models/research/object_detection/images"
At this stage, the number at the end of the xml to csv files
in the relevant repo shows which step will be used for the
dataset to be matched with the experiment. Therefore, the
relevant xml to csv file should be run in the relevant experiment:
python xml_to_csv.py

Except for the first experiment, the same label file containing the pets is used for the rest of the dataset. For the first
step, generate_tfrecord all.py in the repo is used and generate tfrecord pets.py in the repo is used for the others.

python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record

Finally, a label map is created and the training configuration file is edited with relevant paths, making it ready for
training. For these steps: By creating separate training directories, configuration files specific to each experiment were
organized and labelmap.pbtxt file was created. Finally, start
the train by making sure that the inference graph directory
exists and is empty. Run the training: python train.py –
logtostderr –train dir=training/ –pipeline config path =training/faster rcnn inception v2 pets.config At the same time,
the progress of the training job can be tracked using the
TensorBoard.
All label:
item {
id: 1
name: 'aeroplane'
}

item {
id: 2
name: 'bicycle'
}

item {
id: 3
name: 'boat'
}

item {
id: 4
name: 'bus'
}

item {
id: 5
name: 'car'
}

item {
id: 6
name: 'train'
}

item {
id: 7
name: 'bottle'
}

item {
id: 8
name: 'chair'
}

item {
id: 9
name: 'diningtable'
}

item {
id: 10
name: 'pottedplant'
}

item {
id: 11
name: 'motorbike'
}

item {
id: 12
name: 'sofa'
}

item {
id: 13
name: 'tvmonitor'
}

item {
id: 14
name: 'bird'
}

item {
id: 15
name: 'cat'
}

item {
id: 16
name: 'cow'
}

item {
id: 17
name: 'dog'
}

item {
id: 18
name: 'horse'
}

item {
id: 19
name: 'sheep'

}

item {
id: 20
name: 'person'
}




Pets label
item {
id: 1
name: 'bird'
}

item {
id: 2
name: 'cat'
}

item {
id: 3
name: 'cow'
}

item {
id: 4
name: 'dog'
}

item {
id: 5
name: 'horse'
}

item {
id: 6
name: 'sheep'
}

In configuration file:
 num_classes: 6 (for 4 expr) (first exp 20 class)

fine_tune_checkpoint: "C:/tensorflow2/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
train_input_reader: {
  tf_record_input_reader {
    input_path: "C:/tensorflow2/models/research/object_detection/train.record"
  }
  label_map_path: "C:/tensorflow2/models/research/object_detection/training/labelmap.pbtxt"
}
eval_input_reader: {
  tf_record_input_reader {
    input_path: "C:/tensorflow1/models/research/object_detection/test.record"
  }
  label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
  shuffle: false
  num_readers: 1
}
Finally, a label map is created and the training configuration file is edited with relevant paths, making it ready for
training. For these steps: By creating separate training directories, configuration files specific to each experiment were
organized and labelmap.pbtxt file was created. Finally, start
the train by making sure that the inference graph directory
exists and is empty. Run the training:

python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config

At the same time,the progress of the training job can be tracked using theTensorBoard.

!tensorboard --logdir=training --port 8080 --host 127.0.0.1

When the training is completed, the .ckpt file with the highest number in ”model.ckpt-” becomes the new model.After
this stage, the New Trained Object Detection Classifier can
be used and tested.

python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph

Evaluation

python eval.py --logtostderr --checkpoint_dir=inference_graph/ --eval_dir=output_results/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
tensorboard --logdir=output_results --port 8080 --host 127.0.0.1
