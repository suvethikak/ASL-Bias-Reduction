# ASL-Bias-Reduction
Reducing Skin Tone Bias in ASL Detection Computer Vision Models

Title: Seeing What the Model Sees: Grad-CAM-Guided Recognition of Static ASL Fingerspelling Before and After Debiasing

Participants: Emily Ramond, Suvethika Kandasamy Mohana

We both will be working on the baseline model together to understand where our current baseline sits
Emily will be working with Grad-CAM to overlay heatmaps on the images and create metrics for fairness evaluation 
Suvethika will be working on creating the adversarial component of the project 

If a curated dataset is possible, both Emily and Suvethika will be working on curating the dataset with available materials
Once complete, both will work on the report on their respective sections to elaborate on the models

Description of the Objectives: 

We aim to build an image-only classifier for static ASL alphabet recognition and use Grad-CAM to verify the model focuses on the hand rather than background artifacts and diagnose confusions between visually similar handshapes (M vs N, C and O, etc).

Our goals are: 

Examine how this model performs before and after debiasing and examine gender-parity. 

Build a baseline classifier for the ASL alphabet

Verify through Grad-CAM that our model focuses on the hand region rather than the background or skin-related artifacts

Identify common misclassifications 

Evaluate performance by skin-tone group as a proxy for demographic fairness 

Introduce adversarial debiasing using a gradient reversal layer so the model learns skin-tone-invariant feature imbeddings

Compare Grad-CAM explainability and per-group accuracy before vs after debiasing

We expect to see…

> Image Classification of ASL both in image and video classifications are relatively successful, thus we expect the accuracy of our model to be quite high 85%. We expect Grad-CAM to show the truthful hand region of these images. We can compare the Grad-CAM of the baseline model and the adversarial debiased model.

> We hope to identify one or two failures in the model (eg, background leakage, lighting/skin-tone bias, etc) that we can mitigate with a 1-2% accuracy gain. 

Plan: 
Data

Our data will pull from Sign Language MNIST (gray-scale images with 24 letters) for a baseline. 

We will add ASL Alphabet Data, which is RGB with larger images, that represent more realistic backgrounds and varied lighting. 

We may explore the idea of generating our own small curated dataset, but this is only given we have time/materials to do so.

Framework / Tools

We will be using pytorch with TorchVision for model training

ImageNet model, resnet18, mobilenet_v3_small, efficientnet_b0, vit_b_16 are all pre-trained

Grad-CAM uses pytorch-grad-cam for class-discriminative heatmaps, which supports CNN models 

We can use TensorBoard and other basic metric visualizing packages 

Network Architecture

Version 1 - Baseline

ResNet-18 pre-trained model as backbone

Some modifications of images (lighting, cropping)

Pool to make easier to classify, score classification 

Overlay Grad-CAM heatmap on the photo

Version 2 - Adding an Adversarial Model 

Use the same pre-trained ResNet-18 backbone

Add a second head with a Gradient Reversal Layer to predict skin tone, encouraging the model to learn skin-tone-invariant features

Train jointly so the main head classifies ASL signs while the adversary minimizes skin-tone bias

Apply the same image augmentations (lighting, cropping) as Version 1

Use Grad-CAM to confirm focus on hands rather than appearance cues

Inspiration

We are taking inspiration from “Studying and Mitigating Biases in Sign Language Understanding Models” (https://arxiv.org/abs/2410.05206)

For the adversarial debiasing component part of project was inspired by Adversarial Gender Debiasing (Zhao et al., EMNLP 2018) for word embeddings 
(https://www.sciencedirect.com/science/article/pii/S2949719124000402v) 

To our best knowledge, no work has been done with adversarial debiasing and Grad-CAM interpretability specific to ASL image classification 

We will be contributing a new fairness-focused evaluation of these methods. 

This is partial re-implementation, exploring new methodology within a different domain with the potential for a curated dataset 

We selected these models to explore how established techniques can be applied toward a clear goal: reducing bias in computer vision models for American 
Sign Language (ASL) detection. By leveraging frameworks such as Grad-CAM for model interpretability and incorporating an adversarial debiasing component, we aim to evaluate whether tools that have proven effective in other domains—particularly natural language processing (NLP)—can also enhance fairness and transparency in visual recognition systems. This approach allows us to not only assess model accuracy but also analyze and mitigate potential biases in learned visual features, adapting well-known strategies to a new and impactful context.

