MNIST Dataset prediction using Convolutional Neural Network (CNN) model

Inputs: 28x28 grayscale image

<img src="https://github.com/dihcuierc/MNIST-CNN/images/input%20image.png" alt="image sample" width="200"/>

RandomAffine: Random affine transformation of the image

```python
RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1))
```

Model description:

```python
nn.Conv2d(1, 16, kernel_size=5, padding=0, stride=1),  # 1x28x28 --> 16x24x24
nn.BatchNorm2d(16),
nn.ReLU(),
nn.Flatten(),
nn.Linear(9216, 128),
nn.ReLU(),
nn.Dropout(p=0.5),
nn.Linear(128, 10)
```

Outputs: 10 digit classification (0-9):

- Train accracy: 97.4%
- Test accuracy: 99.2%

<img src="https://github.com/dihcuierc/MNIST-CNN/images/accuracy.png" alt="train and test accuracy" width="500"/>

- model weights are saved in MNIST_model.pth

use `model = MNIST(); model.torch.load_state_dict('MNIST_model.pth')` to load the model
