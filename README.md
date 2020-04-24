# A multi-classification model for thoracic disease

## First blood (VGG16_try1.ipynb)

In this file, Zijian has finished the basic data processing of NIH images. For convienent, only 600 images of 14 thoacic desease are adopted to run the model. And he uses the VGG16 model to test the code pipeline.

Next step:
- Increase datasets into 100,000 images
    - Should we delete some rare desease?
    - Add COVID-19 into the datasets
- Deploy the model on high performance computer
- Add other models like, ResNet

## Second try (VGG16_v2.2.ipynb)
In this file, Zijian mainly adjusted the train generator, which allows to read a small number of images at each batch instead of loading all images into the memory.

Besides, Zijian also adjusts the datasets, and mainly loads four desease's images including effusion, atelectasis, infiltration, pneumonia, and also the normal ones.

But the prediction results only contain 3 classes! There might be some problem!
The next step is to solve this problem and adjust the model to achieve a better results.