# ELANet: A Multi-Exposure Fusion Method for Reflection Suppression of Curved Workpieces

## 1. Environment
- Python >= 3.8.8
- PyTorch >= 1.9.0 is recommended
- opencv-python >= 4.5.3
- matplotlib >= 3.3.4
- tensorboard >= 2.7.0
- pytorch_msssim

## 2. Dataset
The training data and testing data is from the CW-MEF dataset.

## 3. Quick Demo
1. Place the over-exposed images and under-exposed images in `dataset/test_data/over` and `dataset/test_data/under`, respectively.
2. Run the following command for multi-exposure fusion:
    ```
    python main.py --test_only
    ```
3. Finally, you can find the fused results in `./test_results`.

## 4. Training and Testing
1. Place the training over-exposed images and under-exposed images in `dataset/train_data/over` and `dataset/train_data/under`, respectively.
2. Run the following command to train your own model:
```
python main.py --model mymodel.pth
```
Or you can fine-tune the existing model based on your own dataset:
```
python main.py --model model.pth
```
Moreover, if you want to test the model after training each epoch, run:
```
python main.py --model mymodel.pth --train_test
```
4. The generated model is placed in `./model/`, then you can test your model with:
```
python main.py --test_only --model mymodel.pth
```
