<<<<<<< HEAD
# Person-Identification-From-Video-Data
=======
# Hand Raised Person Detection and Identification on Video

I have used YOLO model from ultralytics to detect Hand Raised person. Initially, I convert the video into frames. Then annotate the frames with Labelme Tool. Then I convert the labelme .json to yolo .txt format. Then did a yolov8 training with colab.

In the inference, I calculate the center of the bboxes and if the center maps with specific position for 8 different person then show the person name on top of the bbox.

I ignored the bounding box during inference because the person name txt may appear on the bbox somethimes.

## Requirements

Before running the script, you need to install the required dependencies. You can install them using the following command:

```bash
pip install -r requirements.txt
>>>>>>> d8a2019 (Added Readme file for description)
