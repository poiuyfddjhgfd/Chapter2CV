## Chapter 2: Flower Image Classification with TensorFlow/Keras
# 📋 Project Overview
This notebook implements a complete image classification pipeline for flower recognition using TensorFlow and Keras. It demonstrates both simple linear models and more complex neural network architectures to classify flower images into 5 distinct categories.

# 🎯 Objectives
Load and preprocess flower image data

Implement and compare linear vs. neural network models

Visualize training data and results

Train and evaluate classification models

# 📁 Dataset Structure
The project expects a flowers/ directory with the following structure:

text
flowers/
├── class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_2/
│   ├── image1.jpg
│   └── ...
└── ... (3 more classes)
🛠️ Implementation Details
Data Loading & Preprocessing
Image Size: 180×180 pixels (RGB)

Batch Size: 32 images per batch

Train/Validation Split: 80%/20% with fixed random seed

Normalization: Automatic preprocessing via TensorFlow's image_dataset_from_directory

Model Architectures
Linear Model

python
Sequential([
    Flatten(input_shape=(180, 180, 3)),
    Dense(5)  # Output layer for 5 classes
])
Neural Network

python
Sequential([
    Flatten(input_shape=(180, 180, 3)),
    Dense(300, activation="relu"),
    Dense(100, activation="relu"),
    Dense(5)  # Output layer
])
Training Configuration
Optimizer: Adam

Loss Function: Sparse Categorical Crossentropy (with logits)

Metrics: Accuracy

Epochs: 10

# 📊 Features
Data Visualization: Sample images with class labels

Error Handling: Examples of debugging common issues

Modular Design: Clear separation of data loading, model building, and training

Reproducibility: Fixed random seeds for consistent results

# 🚀 Quick Start
Prerequisites
bash
pip install tensorflow matplotlib
Usage
Prepare your dataset: Organize flower images into the flowers/ directory

Run the notebook: Execute cells sequentially

Monitor training: View accuracy and loss metrics during training

Visualize results: See sample images with predictions

Running the Code
python
# Load data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "flowers/",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

# Build and train model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(180, 180, 3)),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(5)
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

history = model.fit(train_ds, validation_data=val_ds, epochs=10)
📈 Expected Results
Training accuracy improvement over epochs

Validation accuracy showing generalization capability

Visual confirmation of correctly classified images

# ⚠️ Common Issues & Solutions
"NameError: name 'train_ds' is not defined"

Ensure you run the data loading cell before model training

Check that the flowers/ directory exists with proper structure

Warning about input_shape in Flatten layer

This is informational; you can safely ignore it

Alternatively, use Input(shape=(180, 180, 3)) as the first layer

Slow training

Reduce image size or batch size

Use data augmentation for better generalization

🔧 Customization
Adjust image size: Modify image_size parameter

Change architecture: Add/remove layers or change neuron counts

Experiment with hyperparameters: Learning rate, batch size, epochs

Add data augmentation: Include preprocessing layers for robustness

# 📚 Learning Outcomes
Through this notebook, you'll learn:

How to load and preprocess image data with TensorFlow

The difference between linear and neural network models

How to compile, train, and evaluate Keras models

Basic debugging techniques for ML pipelines

Best practices for image classification tasks

📄 License
This project is for educational purposes. Feel free to modify and distribute with proper attribution.

# 🤝 Contributing
Suggestions and improvements are welcome! Please open an issue or submit a pull request
