# Part 2 — Computer Vision Problem Formulation and CNN Prototype

## Manufacturing Defect Detection using Convolutional Neural Networks

---

## Problem Statement

Surface defects in manufactured products — scratches, dents, stains — directly affect product quality, customer satisfaction, and brand reputation. Manual visual inspection is slow, inconsistent, and costly at industrial scale. This project builds a CNN-based image classifier that automatically identifies the type of surface defect (or confirms a defect-free surface) from a single product image.

---

## Computer Vision Problem Type

**Multi-class Image Classification**

Each image in the dataset shows a product surface with exactly one label: `normal`, `scratch`, `dent`, or `stain`. There are no bounding boxes and no per-pixel annotations. The task is to predict which of the four categories best describes the surface condition. Object detection or segmentation formulations are unnecessary here because the labels describe the entire image, not regions within it.

---

## Dataset

**Source:** [Shared Google Drive Folder](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing) — Part 2 dataset

| Property | Value |
|---|---|
| Total images | 480 |
| Classes | 4 (normal, scratch, dent, stain) |
| Images per class | 120 (perfectly balanced) |
| Image format | PNG |
| Typical image size | 128 × 128 × 3 |

The dataset is **perfectly balanced** — each class contains exactly 120 images — so no class weighting or oversampling was required.

> **Note:** Dataset files are not committed to this repository per the project guidelines. Download from the Drive link above and place the `images/` folder and `labels.csv` in the same directory as `notebook.ipynb` before running.

---

## Repository Structure

```
part-2-cnn-computer-vision/
│
├── README.md
├── notebook.ipynb               # main notebook: all tasks end-to-end
├── requirements.txt
│
├── sample_predictions/
│   └── prediction_outputs.png   # 12 sample test predictions
│
└── results/
    ├── class_distribution.png   # bar chart of class counts
    ├── sample_images.png        # 4 samples per class
    ├── accuracy_loss_curves.png # training history
    ├── confusion_matrix.png     # test-set confusion matrix
    └── defect_cnn_model.keras   # saved model weights
```

---

## Approach

### Data Split

| Split | Images | Percentage |
|---|---|---|
| Training | 336 | 70% |
| Validation | 72 | 15% |
| Test | 72 | 15% |

Stratified splitting was used to maintain equal class proportions in all three subsets.

### Preprocessing

1. Resize all images to **128 × 128 pixels**
2. Convert to RGB (3 channels)
3. Normalize pixel values to **[0, 1]** by dividing by 255

### Data Augmentation (Training Only)

To improve generalisation and reduce overfitting, the following augmentations were applied during training:

- Random rotation (±15°)
- Horizontal flip
- Width and height shift (±10%)
- Zoom (±10%)

Augmentation was applied only to the training set; validation and test images were used as-is.

---

## CNN Architecture

```
Input (128 × 128 × 3)
│
├── Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU
├── MaxPooling2D(2×2) → Dropout(0.25)         # 64 × 64
│
├── Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU
├── MaxPooling2D(2×2) → Dropout(0.25)         # 32 × 32
│
├── Conv2D(128) → BN → ReLU
├── MaxPooling2D(2×2) → Dropout(0.25)         # 16 × 16
│
├── Flatten
├── Dense(256) → BN → ReLU → Dropout(0.50)
└── Dense(4, softmax)                          # output probabilities
```

**Loss:** Categorical cross-entropy  
**Optimizer:** Adam (lr = 0.001)  
**Callbacks:** EarlyStopping (patience=8), ReduceLROnPlateau (factor=0.5, patience=4)

Three convolutional blocks were chosen to give the model sufficient depth to recognise defect-specific patterns while remaining manageable for a dataset of 480 images. Batch normalisation after each convolutional layer stabilises training. Dropout before the dense layer is the primary regularisation mechanism.

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | ~92–96% (varies by run) |
| Test Loss | ~0.15–0.25 |

The confusion matrix shows that most misclassifications, when they occur, happen between visually similar classes (e.g., *scratch* vs *normal* on pale surfaces). Inter-class confusion between *dent* and *stain* is generally low because their visual patterns differ substantially.

---

## CNN Concept Explanations

### What is Convolution?

Convolution slides a small learnable filter (e.g., 3×3) across the entire image, computing a dot product between the filter and every overlapping patch. The output — a feature map — encodes where in the image a specific pattern (vertical edge, diagonal line, circular curve) is present. Because the same filter is used across all spatial positions, the number of parameters stays small and the model learns position-invariant features.

### Why is Pooling Used?

After convolution, feature maps are spatially large. Max pooling replaces each non-overlapping region with its maximum value, shrinking the spatial dimensions (typically by a factor of 2 in each direction). This reduces the number of computations in subsequent layers and, more importantly, introduces a degree of **translation invariance** — small shifts in defect position within an image do not drastically change the pooled activations. This is valuable in manufacturing because the exact placement of a scratch or dent on the product surface varies from unit to unit.

### Why is ReLU Commonly Used in CNNs?

The Rectified Linear Unit — `f(x) = max(0, x)` — is preferred for three practical reasons:

1. **Gradient flow:** The gradient of ReLU is 1 for positive inputs, so gradients do not shrink as they are backpropagated through many layers (unlike sigmoid or tanh, which saturate and produce near-zero gradients for large inputs).
2. **Computational speed:** A single comparison operation is far cheaper than computing an exponential (sigmoid, softmax).
3. **Sparse activations:** Negative pre-activations produce an output of 0, so only a subset of neurons fire for any given input. This sparsity tends to produce cleaner, more separable representations.

### Why are CNNs Better than Feed-Forward Networks for Image Data?

A fully connected network applied to a 128×128×3 image has 49,152 inputs. Connecting them to even a modest first hidden layer of 512 neurons produces ~25 million parameters — almost certainly leading to severe overfitting on a dataset of 480 images.

CNNs exploit three structural properties that dense networks ignore:

| Property | What it means | Why it matters for images |
|---|---|---|
| Local connectivity | Each neuron looks at a small patch, not the whole image | Nearby pixels are highly correlated; distant pixels are not |
| Weight sharing | The same filter is applied everywhere | A scratch looks similar whether it is in the top-left or bottom-right corner |
| Hierarchical features | Early layers detect edges, later layers detect shapes and objects | Defect classification requires understanding structure at multiple scales |

These inductive biases let a CNN generalise from a small dataset while using orders of magnitude fewer parameters than an equivalent dense network.

---

## Business Use Case: Automated Quality Inspection in Electronics Manufacturing

### The Problem

A smartphone casing manufacturer produces ~3,000 units per shift. Human inspectors can realistically examine around 600–800 casings per shift with sustained attention. Defective units — scratches introduced during machining, dents from handling, stains from lubricants — cost the company in returns and warranty claims when they reach customers.

### The CNN Solution

A 5-megapixel line-scan camera mounted above the conveyor captures an image of each casing. The image is resized, normalised, and passed to the trained CNN. Inference takes less than 10 ms on a mid-range GPU, allowing the line to operate at full speed (>100 units/min). Units classified as `scratch`, `dent`, or `stain` are automatically flagged and diverted to a rework bay; `normal` units proceed to packaging.

### Business Benefits

- **Consistency:** No inspector fatigue; the same decision threshold is applied to every unit across every shift.
- **Coverage:** 100% inspection instead of sampling.
- **Traceability:** Every prediction is logged with the image, timestamp, and machine ID, enabling root-cause analysis when a defect type suddenly spikes.
- **Cost:** Catching defects before packaging eliminates downstream rework, customer returns, and warranty costs that typically cost 5–10× more than in-process detection.

### Extensibility

With denser labelling, the same CNN backbone can be extended to:
- **Object detection** (localise and count multiple defects per image)
- **Severity estimation** (regression on defect area or depth)
- **Process feedback** (correlate defect type frequency with upstream machine parameters)

---

## How to Run

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd part-2-cnn-computer-vision

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset from the Drive link and place
#    the images/ folder and labels.csv here

# 4. Launch the notebook
jupyter notebook notebook.ipynb
```

Run all cells in order. Outputs (plots, confusion matrix, saved model) will be written to `results/` and `sample_predictions/`.

---

## Observations

- The model converges reliably within 20–30 epochs with early stopping.
- Batch normalisation significantly accelerated training compared to a version without it.
- The biggest source of confusion was between `scratch` and `normal` on lighter-coloured surfaces where scratches produce low contrast.
- Augmentation (especially horizontal flips and slight rotations) reduced the gap between training and validation accuracy, indicating it helped with generalisation.

---

## Dependencies

See `requirements.txt`. Key libraries:

| Library | Purpose |
|---|---|
| TensorFlow / Keras | CNN model building and training |
| NumPy | Array operations |
| Pandas | Label CSV loading |
| Pillow | Image loading and resizing |
| scikit-learn | Train-test split, classification report |
| Matplotlib / Seaborn | Visualisation |
