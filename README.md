# Musical Instruments Audio Classification 

In this repository, I experiment with different machine learning algorithms to classify audio samples of musical instruments.

### Data source

The raw audio files, in the `.wav` format, were downloaded from [here](https://github.com/seth814/Audio-Classification/tree/master/wavfiles). 

### Pipeline

1. Load in raw audio file (`.wav`) format. 
2. Sample the signal such that the resultant is discrete data. 
3. Convert to Mel Spectrogram—resultant is an image.  
4. Save resultant image to respective directory. 
5. Repeat for all images. 
6. Run an image classification model on the Mel Spectrogram images. 

The resultant directories that contain the Mel Spectrograms are saved in this repository as well. 

### Repository file structure 
#### Neural network notebooks  
In this repository, several convolutional neural networks are trained and tested. Specifically, the **VGG16**, **VGG19**, and **MobileNet** architectures are used. Each notebook name corresponds to the architecture used. In each of these notebooks, the model is built, trained, and tested. Furthermore, several predictions are displayed for each class. 
