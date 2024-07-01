# Partial Image Restoration Project using EBM (Energy-Based Model)

## Overview
This project focuses on restoring masked images to their original state using an Energy-Based Model (EBM). EBM utilizes contrastive divergence to effectively handle cases where parts of images are missing or damaged, aiming for accurate and smooth restoration.

## Project Goals
- Refactor existing TensorFlow code into PyTorch to implement EBM.
- Train EBM using masked images and evaluate the performance of restored original images.
- Provide an effective image restoration tool, particularly useful for scenarios where parts of images are missing or damaged.

## Features
- **EBM Training:** The model is trained using contrastive divergence. The current implementation of contrastive divergence is undergoing refactoring and is not yet complete.
- **Image Restoration:** Given masked image inputs, EBM predicts missing parts to reconstruct the original image.

## Results from TensorFlow Environment
Here are some example results obtained from the previous TensorFlow implementation:

![image](https://github.com/So1pi/EBMImageCompletion/assets/173986541/593fb036-8106-433e-9909-071836e90d77)

![image](https://github.com/So1pi/EBMImageCompletion/assets/173986541/911850b7-63e3-466d-b08d-5b2069e2deb7)

![image](https://github.com/So1pi/EBMImageCompletion/assets/173986541/a920ec11-0d9f-4391-b19f-fbb93772ca55)




These images showcase the output generated by the TensorFlow implementation before the transition to PyTorch. They demonstrate the initial capabilities of the image restoration process using the Energy-Based Model (EBM).
