# Custom_MNIST_CNN

Custom Convolutional Neural Network (CNN) for experimentation with on the MNIST dataset for open character recognition. Experimenting with architecture and visualization of the model's kernel and weights. Current architecture is one convolutional layer (3x3 kernel), followed by a leaky ReLU activation, max pooling (2x2), and a fully connected layer with softmax that outputs 10 classes for MNIST digit classification

## Results

Best results so far with an 80:20 train:test split over 10 epochs is 92.3% accuracy.

```log
[[1302    1    2    3    1    9   14    2    8    1]
 [   0 1564    5    6    2    5    2    3   11    2]
 [  11   16 1216   27   18    5   25   13   38   11]
 [   3   12   16 1305    0   30    7   13   31   16]
 [   1    2   10    4 1180    8   18    5   11   56]
 [   9   11    6   60   13 1098   27    2   39    8]
 [   4    6    5    1    8   15 1355    0    2    0]
 [   9    7   17   11    9    2    0 1404    3   41]
 [   6   26    7   63    7   28   14    4 1185   17]
 [   7   11    6   22   23    3    0   35   19 1294]]
              precision    recall  f1-score   support

           0       0.96      0.97      0.97      1343
           1       0.94      0.98      0.96      1600
           2       0.94      0.88      0.91      1380
           3       0.87      0.91      0.89      1433
           4       0.94      0.91      0.92      1295
           5       0.91      0.86      0.89      1273
           6       0.93      0.97      0.95      1396
           7       0.95      0.93      0.94      1503
           8       0.88      0.87      0.88      1357
           9       0.89      0.91      0.90      1420

    accuracy                           0.92     14000
   macro avg       0.92      0.92      0.92     14000
weighted avg       0.92      0.92      0.92     14000
```

## Requirements

Python 3.8+, recommended Python 3.10+.

Python packages:

- Numpy
- Matplotlib
- Seaborn
- Scikit-learn

To install the pip requirements, run the following command:

```bash
pip install -r requirements.txt
```

We recommend using a virtual environment to manage dependencies.

## License

This project is freely available, for both personal and commercial use, under the MIT license, see the [LICENSE](LICENSE) file for details.
