# Computer Vision Exam — Cat and Dog Image Classification

A classical computer vision pipeline that classifies images as cat or dog. Images
are preprocessed with OpenCV, flattened into raw pixel vectors, and classified by
a linear Support Vector Machine.

Built as an exam for a Computer Vision course.

## Problem statement

The task was to implement a program that classifies images of cats and dogs using
machine learning, applying the image processing operations covered during the
course rather than a pretrained deep learning model.

## Pipeline

Every image, in both training and test sets, goes through the same four
preprocessing steps:

| Step | Operation | Purpose |
| --- | --- | --- |
| 1 | `cv2.resize` to 128x128 | gives every sample a fixed dimensionality |
| 2 | `cv2.GaussianBlur` with a 5x5 kernel | suppresses high-frequency noise |
| 3 | `cv2.cvtColor` to grayscale | drops colour, keeping shape and texture |
| 4 | `cv2.equalizeHist` | normalizes contrast across differently lit photos |

The resulting 128x128 image is then flattened into a single vector of 16,384
features, which is what the classifier consumes.

## Rationale for the techniques

Histogram equalization and Gaussian smoothing were chosen to reduce the variance
that comes from photos taken under very different lighting conditions, so the
classifier sees more uniform input.

Grayscale conversion cuts the feature count by two thirds and forces the model to
rely on shape and texture instead of coat colour, which is not a reliable
discriminator between the two species.

Flattening is the simplest possible way to represent an image as a feature
vector, with no engineered descriptors involved.

A Support Vector Machine with a linear kernel fits a binary problem well and was
already familiar from previous coursework. With only two classes, a single
separating hyperplane is a natural choice.

Precision, recall and F1-score were used for evaluation because accuracy alone
hides how errors are distributed between the two classes.

## Dataset

```text
Treino/    1000 training images: 455 cats (cat.N.jpg), 545 dogs (dog.N.jpg)
Teste/     6 test images: gato1-3.jpg and cachorro1-3.jpg
```

Labels are derived from the filenames. Training images containing `cat` are
labelled `gato` and those containing `dog` are labelled `cachorro`; test images
follow the same logic on their Portuguese names.

Training and test sets are two fixed, separate folders. There is no random split:
the six test images are listed explicitly in the script.

## Project structure

```text
college-computer-vision-exam/
├── prova.py    # preprocessing, training, prediction and visualization
├── Treino/     # training images
└── Teste/      # test images
```

## Requirements

- Python 3
- opencv-python
- scikit-learn
- matplotlib
- numpy

## Setup

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install opencv-python scikit-learn matplotlib numpy
```

## Running

```bash
python prova.py
```

The script must be run from the repository root, since the image paths
`Treino/` and `Teste/` are resolved relative to the working directory.

It prints the classification report to the terminal and then opens a matplotlib
window with the six preprocessed test images and their true labels.

## Steps performed

1. Load every image from `Treino/` and derive its label from the filename.
2. Preprocess each one: resize, Gaussian blur, grayscale, histogram equalization.
3. Flatten each preprocessed image into a 16,384-dimensional vector.
4. Train an `SVC(kernel='linear')` on the full training set.
5. Apply the identical preprocessing to the six test images.
6. Predict their classes and compare against the true labels.
7. Print precision, recall and F1-score, then display the preprocessed test
   images with their labels.

## Results

```text
=== Métricas de Avaliação ===
              precision    recall  f1-score   support

    cachorro     0.6667    0.6667    0.6667         3
        gato     0.6667    0.6667    0.6667         3

    accuracy                         0.6667         6
   macro avg     0.6667    0.6667    0.6667         6
weighted avg     0.6667    0.6667    0.6667         6
```

The model classified four of the six test images correctly, with errors split
evenly between the two classes.

Two factors bound this result. A test set of six images makes the metrics coarse,
since every single image moves accuracy by roughly 17 percentage points. And a
linear SVM over raw pixel intensities has no notion of spatial structure: two
photographs of the same animal at different positions or scales produce entirely
different feature vectors. Closing the gap would require either engineered
descriptors such as HOG, or a convolutional model.

## Time spent

Roughly 1 hour and 40 minutes, including development and documentation.

## Difficulties encountered

Obtaining a usable training dataset was the main obstacle. The library suggested
for downloading it did not work, so the dataset had to be sourced separately and
added to the repository manually.
