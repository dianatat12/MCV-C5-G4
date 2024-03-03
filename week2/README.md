# C5-Project-Week 2 Object Detection, recognition and segmentation

We are group 4, composed of:
- Aleix Pujol - aleixpujolcv@gmail.com
- Diana Tat - dianatat120@gmail.com
- Georg Herodes - georgherodes99@gmail.com
- Gunjan Paul - gunjan.mtbpaul@gmail.com

Link to the presentation: https://docs.google.com/presentation/d/1nzJLR1mX-PKBNm1XclOPc5sPGekgsm1s-84SitKpzbw/edit#slide=id.g2be03bd668c_0_20

Link to the report: https://www.overleaf.com/9918637622cndkjfsrhqrf#03d873

### Week 2 Tasks:
#### Task (a): Get familiar with Detectron2 framework
- Install the Detectron2 framework.
- Follow the Detectron2 beginner’s tutorial.

#### Task (b): Set up project
- Familiarize yourself with reading images and annotations.
- Find the KITTI-MOTS dataset in the server directory: `/home/mcv/datasets/C5/KITTI-MOTS/`.

#### Task (c): Run inference with pre-trained Faster R-CNN and Mask R-CNN on KITTI-MOTS dataset
- Apply Faster R-CNN and Mask R-CNN using Detectron2 framework with pretrained COCO weights.
- Project presentation includes dataset description and qualitative results.

#### Task (d): Evaluate pre-trained Faster R-CNN and Mask R-CNN on KITTI-MOTS dataset
- Use official validation partition of KITTI-MOTS as the test set.
- Don’t use KITTI-MOTS evaluation metrics; instead, use official COCO metrics provided by Detectron2.
- Modify MetadataCatalog for class label mapping.
- Project presentation includes metric description and quantitative results.

#### Task (e): Fine-tune Faster R-CNN and Mask R-CNN on KITTI-MOTS
- Train Faster R-CNN and Mask R-CNN using Detectron2 framework on the KITTI-MOTS dataset.
- Split the original training set into training and validation sets.
- Evaluate fine-tuned models on the test set using COCO metrics.
- Compare results with pre-trained models without fine-tuning.
- Project presentation includes quantitative and qualitative results.

#### [OPTIONAL] Task (f.1): Apply some other object detection model on KITTI-MOTS : Yolo v9
- Train another object detection model on the KITTI-MOTS dataset.
- Evaluate the model on the test set and compare results with Faster R-CNN and Mask R-CNN.




