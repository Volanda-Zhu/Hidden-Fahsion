# Hidden-Fashion
This project  is to help improve the performance of fashion image analysis in real-world applications. The images are collected from both commercial stores and customers and contains 491K images of 13 popular clothing categories with rich annotations. By training the model on pairs of commercial and customer's photos, we bridge the gap and make the model more commercially applicable.

Our goal is to help improve performance of fashion image analysis in real-world applications.Based on this, our
project mainly focus on three tasks:

1) Landmark Estimation: This task aims to predict landmarks for each detected clothing item. AP pt is employed
as the evaluation metrics. In the DeepFashion2 dataset, landmark annotations are provided and our first task is to
predict the landmarks and evaluate it by Average Precision (AP).

2) Category Classification: This task predicts the clothes categories by clothes in an image by predicting
bounding boxes and category labels.

3) Clothes Retrieval: The task of clothes retrieval is to correctly detect the clothes in the commercial picture from
shop or a customer picture from street and make semantic segmentation. Our goal is to establish a model that
can identify content of the images, especially for two or more basic visual-related products simultaneously. The
semantic fashion style and function should be correctly classified. And then potentially the segmentation would
be used for comparing and discovering the similarity among products. This will be discussed in the latter part.
