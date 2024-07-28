# bharat_intern_IRIS_CLASSIFICATION.


Project : IRIS_CLASSIFICATION

The project aims to classify the species of the Iris flower using a neural network implemented with Keras. The dataset used is the famous Iris dataset, which includes features such as sepal length, sepal width, petal length, and petal width.

Tools and Technologies Implemented
Python: The primary programming language used for the project.
Pandas: For data manipulation and analysis.
NumPy: For numerical operations and handling arrays.
Seaborn: For data visualization.
TensorFlow/Keras: For building and training the neural network model.
Scikit-learn: For preprocessing the data, such as encoding labels, splitting the dataset, and scaling features.
Concepts Used
Data Loading and Exploration:

The Iris dataset is loaded using Pandas.
Basic data exploration techniques are used, such as viewing the first few rows (df_iris.head()), checking the distribution of species (df_iris['Species'].value_counts()), and checking for missing values (df_iris.isnull().sum()).
Label Encoding:

The Species column, which contains categorical labels, is encoded into numeric values using LabelEncoder.
Data Splitting:

The dataset is split into training and testing sets using train_test_split from Scikit-learn, with 70% of the data used for training and 30% for testing.
Feature Scaling:

Features are scaled using StandardScaler to ensure they have a mean of 0 and a standard deviation of 1. This helps in speeding up the convergence of the neural network.
One-Hot Encoding:

The target variable y_train is converted into a one-hot encoded format using keras.utils.to_categorical for training the neural network with a categorical cross-entropy loss function.
Neural Network Model:

A Sequential model is built using Keras with the following layers:
An Input layer to define the shape of the input data.
Two hidden Dense layers with 32 units and ReLU activation.
A Dropout layer with a rate of 0.5 to prevent overfitting.
An output Dense layer with 3 units (one for each class) and softmax activation.
Model Compilation and Training:

The model is compiled with the Adam optimizer and categorical cross-entropy loss.
The model is trained for 100 epochs.
Model Prediction:

Predictions are made on the test set using the trained model.
The predicted probabilities are converted to class labels using np.argmax.
Model Evaluation:

The accuracy of the model is evaluated using accuracy_score from Scikit-learn.
A confusion matrix is generated and visualized using Seaborn to understand the performance of the model.
Insights
Data Preprocessing: Proper preprocessing, such as scaling features and one-hot encoding labels, is crucial for the performance of neural networks.
Model Architecture: A simple neural network with a few hidden layers and dropout can effectively classify the Iris dataset.
Model Evaluation: The confusion matrix provides a detailed view of the model's performance, highlighting where it performs well and where it may be making mistakes.
Visualization: Data visualization, such as plotting the confusion matrix, helps in better understanding and interpreting the model's performance.
