# Fingerprint Matching with a Siamese Network
 
Fingerprint **verification** (1:1 matching) using a Siamese neural network with a
shared-weight CNN feature extractor. Given two fingerprint images, the model predicts
whether they belong to the same finger.
 
The project also runs a small **ablation** comparing two ways of merging the paired
feature vectors, which turns out to be the single most important design choice.
 
---
 
## Results
 
Both variants share the same CNN encoder and are trained for 15 epochs with
binary cross-entropy on labeled image pairs from SOCOFing. The only difference is how
the two feature maps are combined before the classifier head.
 
| Merge strategy            | Best validation accuracy | Notes                                              |
| ------------------------- | :----------------------: | -------------------------------------------------- |
| **Element-wise subtract** |        **97.7%**         | Stable; validation loss tracks training loss       |
| L1 distance (`abs`)       |          ~57%            | Fails to generalize (val accuracy near chance)     |
 
The subtract-merge keeps spatial structure (it merges feature *maps*, then applies a
further conv block before flattening), while the L1 variant flattens first and collapses
spatial information into a flat distance vector. On this dataset the former generalizes
and the latter does not.
 
---
 
## Architecture
 
```
        Image A (96x96x1)        Image B (96x96x1)
              |                        |
              +-----> shared CNN <-----+        (Conv32 -> MaxPool -> Conv32 -> MaxPool)
              |                        |
           feat A                   feat B
              \                        /
               \----- merge ---------/           subtract  (variant 1)
                        |                         |abs diff| (variant 2)
                   classifier head               (Conv/Flatten -> Dense64 -> Dense1, sigmoid)
                        |
                  match / no-match
```
 
- **Encoder:** two `Conv2D(32, 3x3) -> MaxPool` blocks, weights shared across both inputs.
- **Head:** dense layers ending in a single sigmoid unit.
- **Loss / optimizer:** binary cross-entropy, Adam.
- **Inference:** trained model is also exported to **TFLite** for lightweight deployment,
  and per-match similarity-scoring latency is profiled in the notebook.
> This is a fully learned, image-based approach. It does **not** use hand-crafted
> minutiae features (ridge endings / bifurcations); the CNN learns its own
> representation directly from pixels.
 
---
 
## Dataset
 
[**SOCOFing**](https://www.kaggle.com/datasets/ruizgara/socofing) (Sokoto Coventry
Fingerprint Dataset): real fingerprints plus synthetically *altered* versions at three
difficulty levels (Easy / Medium / Hard). Images are read as grayscale and resized to
`96x96`. Training pairs are generated on the fly by a `keras.utils.Sequence` data
generator that mixes genuine (same-finger) and impostor (different-finger) pairs.
 
Augmentation (via `imgaug`): Gaussian blur, affine scaling, translation, and rotation.
 
---
 
## Repository structure
 
```
.
├── data.py                     # load / organize the SOCOFing arrays
├── Preprocess.ipynb            # grayscale read, resize, array prep
├── train.ipynb                 # pair generator, both model variants, training + eval
├── datasets/                   # .npy image/label arrays (real + altered)
├── Models/H5/                  # saved Keras models
├── Data Augmentation samples/  # example augmented images
├── Diagrams/                   # architecture / result figures
└── requirements.txt
```
 
---
 
## Setup
 
```bash
git clone https://github.com/OmGhag/FingerPrintMatchingSNN.git
cd FingerPrintMatchingSNN
 
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
 
Key libraries: TensorFlow / Keras, NumPy, OpenCV, imgaug, Matplotlib.
 
## Usage
 
1. **Prepare data** — load and organize the fingerprint arrays:
```bash
   python data.py
```
2. **Preprocess** — run `Preprocess.ipynb` (grayscale read, resize, array prep).
3. **Train & evaluate** — run `train.ipynb`. It builds the pair generator, trains both
   merge variants, plots train/validation curves, profiles matching latency, and exports
   the TFLite model.
---
 
## Limitations & next steps
 
- Reported numbers are **validation** accuracy from the training split; a held-out,
  identity-disjoint test set would give a stronger generalization estimate.
- Verification metrics beyond accuracy (EER, ROC / AUC, FAR–FRR at a fixed threshold)
  would better characterize a biometric matcher and are a natural addition.
- The pipeline is image-based; a minutiae-aware variant could be compared as future work.
