# Seeing What the Model Sees: Grad-CAM-Guided Recognition of Static ASL Fingerspelling Before and After Debiasing

## Team Members
- **Emily Ramond** 
- **Suvethika Kandasamy Mohana**

## Division of Work
- Both team members will work together on the baseline model to understand current performance
- **Emily**: Grad-CAM implementation, overlay heatmaps on images, create metrics for fairness evaluation
- **Suvethika**: Adversarial debiasing component
- Both will collaborate on dataset curation (if feasible) and report writing for their respective sections

## Objectives

We aim to build an image-only classifier for static ASL alphabet recognition and use Grad-CAM to verify the model focuses on the hand rather than background artifacts and diagnose confusions between visually similar handshapes (M vs N, C and O, etc).

### Goals
1. Examine model performance before and after debiasing and examine gender-parity
2. Build a baseline classifier for the ASL alphabet
3. Verify through Grad-CAM that our model focuses on the hand region rather than the background or skin-related artifacts
4. Identify common misclassifications
5. Evaluate performance by skin-tone group as a proxy for demographic fairness
6. Introduce adversarial debiasing using a gradient reversal layer so the model learns skin-tone-invariant feature embeddings
7. Compare Grad-CAM explainability and per-group accuracy before vs after debiasing

### Expected Results
- Image classification of ASL (both image and video) has been relatively successful, thus we expect the accuracy of our model to be quite high (>85%)
- We expect Grad-CAM to show the truthful hand region of these images
- We can compare the Grad-CAM of the baseline model and the adversarial debiased model
- We hope to identify one or two failures in the model (e.g., background leakage, lighting/skin-tone bias) that we can mitigate with a 1-2% accuracy gain

## Implementation Plan

### Data
- **Sign Language MNIST**: Gray-scale images with 24 letters for baseline
- **ASL Alphabet Data**: RGB with larger images, representing more realistic backgrounds and varied lighting
- Potential small curated dataset (time and materials permitting)

### Framework & Tools
- **PyTorch** with TorchVision for model training
- **Pre-trained models**: ResNet-18, MobileNet v3 Small, EfficientNet-B0, ViT-B/16
- **pytorch-grad-cam**: For class-discriminative heatmaps (supports CNN models)
- **TensorBoard**: For metric visualization

### Network Architecture

#### Version 1 - Baseline
- ResNet-18 pre-trained model as backbone
- Image modifications (lighting, cropping)
- Pooling for easier classification
- Score classification
- Overlay Grad-CAM heatmap on photos

#### Version 2 - Adversarial Debiasing
- Same pre-trained ResNet-18 backbone
- Additional head with Gradient Reversal Layer to predict skin tone
- Encourages learning of skin-tone-invariant features
- Joint training: main head classifies ASL signs while adversary minimizes skin-tone bias
- Same image augmentations as Version 1
- Grad-CAM verification of focus on hands rather than appearance cues

## Inspiration & Related Work

This project draws inspiration from:
- ["Studying and Mitigating Biases in Sign Language Understanding Models"](https://arxiv.org/abs/2410.05206)
- Adversarial debiasing techniques from ["Adversarial Gender Debiasing" (Zhao et al., EMNLP 2018)](https://www.sciencedirect.com/science/article/pii/S2949719124000402v) for word embeddings

To our best knowledge, no work has been done combining adversarial debiasing and Grad-CAM interpretability specifically for ASL image classification. We contribute a new fairness-focused evaluation of these methods.

This is a partial re-implementation exploring new methodology within a different domain with the potential for a curated dataset.

## Motivation

We selected these models to explore how established techniques can be applied toward a clear goal: reducing bias in computer vision models for American Sign Language (ASL) detection. By leveraging frameworks such as Grad-CAM for model interpretability and incorporating an adversarial debiasing component, we aim to evaluate whether tools that have proven effective in other domains—particularly natural language processing (NLP)—can also enhance fairness and transparency in visual recognition systems. This approach allows us to not only assess model accuracy but also analyze and mitigate potential biases in learned visual features, adapting well-known strategies to a new and impactful context.
