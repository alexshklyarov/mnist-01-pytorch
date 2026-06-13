# MNIST Digit Classifier (0 vs 1) — Neural Network from Scratch in PyTorch

A binary classifier that distinguishes handwritten **0s** from **1s**, built from scratch in
PyTorch with a manually written training loop. Trained and evaluated on the real MNIST
dataset, reaching **99.95% test accuracy** on ~2100 held-out images.

This is a from-scratch reimplementation of the digit-recognition lab from Andrew Ng's
*Machine Learning Specialization* (originally TensorFlow/Keras), rebuilt in **PyTorch** with
every step reasoned through manually — data loading, normalization, the network, the training
loop, evaluation, and visualizations.

## What it does

- Loads the **MNIST** dataset via `torchvision` and filters it down to the digits 0 and 1
  (~12,665 training images, ~2,115 test images).
- Flattens each 28x28 image into a 784-dimensional vector and scales pixel values to `[0, 1]`.
- Defines a small feed-forward network: `784 -> 16 (ReLU) -> 1 (sigmoid)`.
- Trains it with a **hand-written training loop** (forward -> loss -> backward -> step), using
  binary cross-entropy loss and the Adam optimizer.
- Evaluates accuracy on the official MNIST test split (images the model never saw).
- Performs **error analysis** — inspects the misclassified image (a malformed, slanted 0 that
  looks like a 1 even to a human).
- Visualizes the **learned weight templates** of all 16 hidden neurons, showing how some
  specialize into ring-shaped "0 detectors" and others into vertical-stroke "1 detectors" —
  with no manual guidance, purely from gradient descent.

## Results

- **Test accuracy: 99.95%** (1 misclassification out of ~2,115 test images).
- The single error is an ambiguous, heavily slanted 0 that resembles a 1.
- Loss converges within ~100 epochs, thanks to the large, clean dataset.

## Architecture

| Layer | Type | Shape | Params |
|-------|------|-------|--------|
| Hidden | Linear + ReLU | 784 -> 16 | 12,560 |
| Output | Linear + Sigmoid | 16 -> 1 | 17 |
| **Total** | | | **12,577** |

## Tech stack

- Python, PyTorch, torchvision
- NumPy, Matplotlib

## How to run

```bash
git clone https://github.com/YOUR_USERNAME/mnist-01-pytorch.git
cd mnist-01-pytorch
pip install torch torchvision numpy matplotlib
jupyter notebook mnist.ipynb
```

The MNIST dataset downloads automatically on the first run.

## What I learned

- Loading and filtering a real image dataset with `torchvision`.
- Why images are flattened (784 inputs) for a fully-connected network, and what geometric
  information is lost in the process.
- The difference between scaling (`/255`) and Z-score normalization, and when each is appropriate.
- Why **ReLU** is used in hidden layers (avoids vanishing gradients) while **sigmoid** stays
  on the output (produces a probability).
- How to do **error analysis** and read a model's mistakes as data quality signals.
- How hidden neurons self-organize into feature detectors — visualized directly from their weights.
