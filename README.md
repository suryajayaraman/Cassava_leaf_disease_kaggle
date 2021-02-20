# Cassava_leaf_disease_kaggle

Repo for the Cassave Leaf disease detection kaggle competition. Total List of experiments 
and their results are tabulated in the Model_versioning.xlsx sheet

# Competition results

- Final solution got 914 on Private LB. The solution consisted of ensemble of 5 models
  weighted by confusion matrix values and 4xTTA

- The same solution without TTA would have got 440 on Private LB, unfortunately 
  the solution wasn't selected for final submission


# Final solution description

- Only 2020 competition data was used, created an npy format dataset instead of jpg 
  files to boost loading time, but con is it consumes ~10 times more space

- CV consists of Stratified 5-fold and average of 5folds was taken as cv score

- Final solution consists of ensemble of 5 models - 2 se_resnext50 models, 1 ViT, 
  1 tf_efficientnet_b3_ns and tf_efficientnet_b4_ns model

- Predictions of 5fold of each model was averaged. TTA may or may not be applied.

- Each Models prediction was scaled by confusion matrix.
  model_prediction = np.matmul(confusion_matrix * avg_prediction)

- Output of above step was averaged


## Loss functions
- nn.CrossEntropy, Bi-tempered logistic loss with labelsmoothing and labelsmoothingCE
  loss functions were used

## Optimizer & LR schedulers

- Adam optimizer with weight decay was used

- Cosine annealing w or w/o restarts, 1cycle LR and cyclic LR schedulers were tried


## Augumentations used

- Standard augumentations - Resized to 512 x 512, Flips, rotation, HSV, shiftscale rotate, 
  Gaussian noise, median and Gaussian blurs, Finally normalized by imagenet means and std 

- fmix and cutmix augumentation were applied 

## Other learnings

- AMP, Automatic mixed precision was used 

- custom lr_find function (based on lr range test) was used to set the initial LR

- Learnt to setup TPU with pytorch on kaggle platform in a ViT based model

- Meta Learning (learning Ensembling weights) using OOF prediction of each model, 
  didnt work out correctly due to incorrect OOF Dataset


##  Things not tried

- Stacking / other methods of learning ensembling weights

- Pseudo labelling / distillation learing

- Hyperparameter optimization - took lot of time experimenting, need to find smarter ways 
  to experiment

- Using GradCAM for error improvement, Visualising attention maps of ViT etc



