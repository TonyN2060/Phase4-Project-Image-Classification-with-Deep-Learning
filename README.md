# **Phase4-Project-Image-Classification-with-Deep-Learning**
### **Authors**
- Wambui Thuku
- Wilfred Njagi
- Muthoni Kahura
- Kenneth Kimani
- Cynthia Nasimiyu
- Tony Munene
- Bryan Okwach


## **Business Understanding**

**Project Objective:**
The primary objective of this project is to develop a deep neural network model that can accurately classify whether a pediatric patient has pneumonia or not, based on chest X-ray images. This project aims to showcase the practical application of deep learning in the medical domain, specifically in diagnosing pneumonia using medical images. The project is focused on achieving a proof of concept and demonstrating the ability to iterate and improve the model's performance.

**Business Problem:**
Pneumonia is a significant health concern among pediatric patients, and early diagnosis is crucial for effective treatment. However, diagnosing pneumonia through traditional methods can be time-consuming and subject to human error. The use of deep learning and image classification techniques has the potential to expedite and enhance the accuracy of pneumonia diagnosis, thereby improving patient outcomes.

## **Data Understanding**
The dataset consists of chest X-ray images (anterior-posterior) from pediatric patients aged one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. The dataset is organized into three main folders:

_Train:_ Used for training the model.

_Test:_ Used for evaluating the model's performance.

_Validation(Val):_ Used for fine-tuning and validating the model during the training process.

Within these folders, there are subfolders for each image category (Pneumonia/Normal), summing up to a total of 5,863 X-ray images. The images were screened for quality control, and the diagnoses were graded by two expert physicians. A third expert also checked the evaluation set to account for any grading errors.

### **EDA**
Before building the model, the data underwent several preprocessing steps:
- _Data Distribution Analysis:_ The distribution of the classes in the dataset was visualized using bar plots to understand the balance between the categories.
- _Sample Image Visualization:_ Sample images from the dataset were displayed to provide a visual sense of the data and its characteristics.
- _Image Dimensions:_ Checking the dimensions of the images to ensure they are consistent and suitable for the model.
- _Pixel Intensity Distribution:_ Analyzing the pixel intensity distribution to check if there are any variations or anomalies.

## **Data Preparation**
### **Data Preprocessing**
- The images were loaded in grayscale.
- _ Data Augmentation:_ Data augmentation techniques were applied to diversify the training data. This includes techniques like resizing, rotation, and flipping.
- _Resizing/Rescaling:_ The pixel values of the images were normalized to the range [0, 1] using the rescaling factor (1/225).
- _Handling Class Imbalance:_ Use of class weights to give more importance to the underrepresented class.
- The ImageDataGenerator from Keras was used for image data augmentation and preprocessing, which also facilitates feeding data into the model during training.
  
## **Modelling**
### **Model 1 - Baseline Convolutional Neural Network (CNN) model**
The model is used for binary image classification. The architecture comprises convolutional layers with increasing filters, followed by max-pooling layers to capture features. Flattened outputs are fed through dense layers, eventually leading to a two-unit softmax layer for classification. The model is compiled with Sparse Categorical cross-entropy loss and trained for five epochs using training and validation data, enabling it to learn and enhance its classification capabilities over time. 
The original model's test accuracy of 0.74 and test loss of 0.9644762277603149 might have been influenced by the following factors:
- _Model Complexity:_ The original model's architecture might not have been complex enough to capture intricate patterns within the data, leading to suboptimal performance.
- _Overfitting:_ Without regularization techniques like dropout, the original model could have overfitted the training data, resulting in reduced generalization ability and higher test loss.


### **Model 2 - CNN with Dropout**
The model architecture is designed with a sequence of convolutional layers followed by max-pooling layers to capture intricate features from input images. In order to counter overfitting, dropout layers are strategically inserted after the convolutional layers, which randomly deactivate a subset of neurons during each training epoch. The architecture concludes with fully connected (dense) layers featuring dropout regularization and culminates in an output layer equipped with softmax activation for binary classification. The implementation employs the early stopping technique, which monitors validation loss and halts training if improvement stagnates for a specified number of epochs (patience). By combining dropout and early stopping, the code aims to fortify the model's ability to generalize effectively, enhancing its performance while guarding against overfitting.
The model with dropout exhibited improvements with a test accuracy of 0.807443380355835 and a test loss of 0.2814374566078186. These variations might be attributed to:
- _Regularization:_ The inclusion of dropout layers helps prevent overfitting by introducing randomness during training. While the test accuracy improved, it's possible that further hyperparameter tuning could enhance the model's performance.
- _Reduced Overfitting:_ Dropout's regularization effect could have aided the model in generalizing better to unseen data, leading to the observed reduction in test loss.

### **Model 3 - Transfer Learning through the VGG16**
This model demonstrates transfer learning using the VGG16 architecture, a pre-trained deep neural network model. By leveraging knowledge from ImageNet, the model is fine-tuned for a new image classification task. The base VGG16 layers are frozen, allowing the model to focus on learning task-specific features. The architecture is extended with global average pooling and dense layers for prediction. An early stopping callback halts training if validation loss stalls, preventing overfitting. This approach enhances performance by capitalizing on pre-existing knowledge while adapting to the new classification task.
The model with transfer learning from a pre-trained base model yielded impressive results with a test accuracy of 0.88511 and a test loss of 0.2814374566078186:
- _Feature Extraction:_ Leveraging the pre-trained features of the VGG16 model enabled the model to capture rich and relevant image features, enhancing its ability to discriminate between classes.
- _Generalization:_ The base model's pre-trained features were likely learned from a vast dataset, enabling the model to generalize well to the new task, resulting in a significant increase in test accuracy.


In summary, variations in test results can be attributed to differences in model architectures, regularization techniques, and the use of pre-trained features. Careful consideration of these factors and potential hyperparameter tuning can lead to improved model performance.


## **Recommendations**
The developed deep learning model will offer several advantages to the healthcare industry:

**Efficient Diagnosis:** The model can rapidly process and analyze chest X-ray images to provide a binary diagnosis (pneumonia or non-pneumonia). This speed can lead to quicker decisions and treatment initiation.

**Accuracy Improvement:** The model has the ability to learn complex patterns and features in images that might not be easily discernible by human experts. This could lead to more accurate diagnoses.

**Reduction of Human Error:** By automating the diagnosis process, the model can help reduce the likelihood of human errors that occur in the manual interpretation of medical images.

**Scalability:** The model can be used to analyze a large number of images quickly and consistently, making it suitable for high-throughput scenarios.

