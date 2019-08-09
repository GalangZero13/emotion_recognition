# Emotion Recognition
Simple Emotion Recognition for internship project.

## Requirements (What to check before you do)
Make sure you already have the following things installed:
- Windows 8~10 or Linux Ubuntu (I'm using 18.04)
- Numpy
- Scipy
- OpenCV 3 or 4
- Scikit-learn (if you want to compare it to this library usage)
- Keras (with Tensorflow Backend)
- Anaconda 3
- h5py
- Matplotlib
- Json
- Webcam or Video File to do the real time programming

## How to Run
### Emotion Detector
- If python 3, use `python3` instead
```bash
python -c haarcascade_frontalface_default.xml \
-m checkpoints/epoch_*num*.hdf5
```
- If you wish to use a video file instead of webcam add ```-v video_path/video_file.format``` at the end of the above run text

- In my laptop, the internal webcam uses ```cv2.VideoCapture(0)``` or ```cv2.VideoCapture(1)``` while external webcam uses ```cv2.VideoCapture(2)```

### Train Recognizer
- Run
```bash
python train_recognizer.py -c checkpoints \
-m checkpoints/epoch_*num*.hdf5 \
-s *num epoch to restart at*
```
or 
```bash
python train_recognizer.py --checkpoints checkpoints \
--model checkpoints/epoch_*num*.hdf5 \
--start-epoch *num epoch to restart at*
```


### Test Recognizer 
- Run
```bash
python test_recognizer.py -m checkpoints/epoch_*num*.hdf5
```
or
```bash
python test_recognizer.py --model checkpoints/epoch_*num*.hdf5
```


## Acknowledgment
This project is the codification of the module "Deep Learning for Computer Vision with Python" Chapter 1~3 by Dr. Adrian Rosenberck.
Great great thanks to him for making it easy for me to learn DLCV.
There are some module that were not included in the module, so I kinda interpret it myself. Thank goodness it works with the remaining module.

