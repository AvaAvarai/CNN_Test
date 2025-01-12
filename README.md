# Custom_MNIST_CNN

Custom Convolutional Neural Network (CNN) for experimentation with on the MNIST dataset for open character recognition. Experimenting with architecture and visualization of the model's kernel and weights. Current architecture is one convolutional layer (3x3 kernel), followed by a leaky ReLU activation, max pooling (2x2), and a fully connected layer with softmax that outputs 10 classes for MNIST digit classification

## Results

Best results so far with an 80:20 train:test split over 10 epochs is 92.06% accuracy.

```txt
[[1302    1    3    1    1    7   15    3   10    0]
 [   0 1563    6    9    1    7    1    2    9    2]
 [  11   15 1236   24   17    4   25   10   30    8]
 [   6    9   23 1298    0   42    6    9   26   14]
 [   1    2   11    6 1167    6   18    6   14   64]
 [  11   12    6   50   13 1112   24    4   35    6]
 [   6    3    7    1   10   12 1354    0    3    0]
 [   6   11   18    7    9    2    0 1392    9   49]
 [   7   30   12   64    6   32   14    5 1180    7]
 [   8   11    7   24   35    8    0   36   15 1276]]
              precision    recall  f1-score   support

           0       0.96      0.97      0.96      1343
           1       0.94      0.98      0.96      1600
           2       0.93      0.90      0.91      1380
           3       0.87      0.91      0.89      1433
           4       0.93      0.90      0.91      1295
           5       0.90      0.87      0.89      1273
           6       0.93      0.97      0.95      1396
           7       0.95      0.93      0.94      1503
           8       0.89      0.87      0.88      1357
           9       0.89      0.90      0.90      1420

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
