
// Notebook data embedded to avoid CORS/Fetch issues on file:// protocol
window.MNIST_NOTEBOOK_DATA = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# MNIST Digit Recognition (Browser Version)\n",
                "\n",
                "> **Note:** This notebook has been adapted to run entirely in your browser using `scikit-learn`. The original TensorFlow code has been translated to equivalent Scikit-Learn models (`MLPClassifier`) to ensure real-time performance without requiring Python backend installation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "import microsite\n",
                "await microsite.install(['scikit-learn', 'numpy', 'matplotlib', 'seaborn'])\n",
                "\n",
                "# Import libraries\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.neural_network import MLPClassifier\n",
                "from sklearn.metrics import accuracy_score, confusion_matrix\n",
                "from sklearn.datasets import load_digits\n",
                "from sklearn.model_selection import train_test_split\n",
                "\n",
                "print(\"Libraries loaded successfully!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load Data (Using scikit-learn's digits dataset for browser speed)\n",
                "# Note: These are 8x8 images, optimized for instant browser training\n",
                "digits = load_digits()\n",
                "x_data, y_data = digits.data, digits.target\n",
                "\n",
                "# Normalize data (0-16 -> 0-1)\n",
                "x_data = x_data / 16.0\n",
                "\n",
                "# Split into Train/Test\n",
                "x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)\n",
                "\n",
                "print(f\"Data Loaded: {x_train.shape[0]} training samples, {x_test.shape[0]} test samples\")\n",
                "\n",
                "# Visualize some samples\n",
                "plt.figure(figsize=(10, 4))\n",
                "for index, (image, label) in enumerate(zip(x_train[:5], y_train[:5])):\n",
                "    plt.subplot(1, 5, index + 1)\n",
                "    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)\n",
                "    plt.title('Training: %i' % label)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Training Baseline Model (Logistic Regression)...\")\n",
                "baseline = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')\n",
                "baseline.fit(x_train, y_train)\n",
                "baseline_acc = accuracy_score(y_test, baseline.predict(x_test))\n",
                "print(f\"Baseline Accuracy: {baseline_acc * 100:.2f}%\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"\\nTraining MLP (Neural Network)...\")\n",
                "# Simulating a lighter architecture: Dense(64) -> Dense(32) -> Output(10)\n",
                "# Note: Reduced size to prevent browser UI freezing\n",
                "model = MLPClassifier(hidden_layer_sizes=(64, 32), \n",
                "                      activation='relu', \n",
                "                      solver='adam', \n",
                "                      max_iter=250,\n",
                "                      random_state=42)\n",
                "\n",
                "model.fit(x_train, y_train)\n",
                "\n",
                "test_acc = model.score(x_test, y_test)\n",
                "print(f\"\\nFinal Results:\")\n",
                "print(f\"Baseline Accuracy: {baseline_acc * 100:.2f}%\")\n",
                "print(f\"MLP Test Accuracy: {test_acc * 100:.2f}%\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Confusion Matrix Visualization\n",
                "y_pred = model.predict(x_test)\n",
                "cm = confusion_matrix(y_test, y_pred)\n",
                "\n",
                "plt.figure(figsize=(8,6))\n",
                "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\n",
                "plt.title('MLP Confusion Matrix')\n",
                "plt.xlabel('Predicted Digit')\n",
                "plt.ylabel('Actual Digit')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Loss Curve\n",
                "plt.figure(figsize=(8,5))\n",
                "plt.plot(model.loss_curve_)\n",
                "plt.title('MLP Training Loss over Iterations')\n",
                "plt.xlabel('Iterations')\n",
                "plt.ylabel('Loss')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualization of Predictions\n",
                "import random\n",
                "num_samples = 5\n",
                "indices = random.sample(range(len(x_test)), num_samples)\n",
                "\n",
                "plt.figure(figsize=(15, 5))\n",
                "for i, idx in enumerate(indices):\n",
                "    # Get image and true label\n",
                "    img = x_test[idx]\n",
                "    true_label = y_test[idx]\n",
                "    \n",
                "    # Reshape for prediction (1, 64) for sklearn digits\n",
                "    # Note: Scikit-learn digits are already flattened/ready, but we check shape\n",
                "    # img is (64,) array in this dataset, reshaping to (1, 64)\n",
                "    prediction_probs = model.predict_proba(img.reshape(1, -1))\n",
                "    predicted_label = np.argmax(prediction_probs)\n",
                "    confidence = np.max(prediction_probs) * 100\n",
                "    \n",
                "    plt.subplot(1, num_samples, i + 1)\n",
                "    # Reshape back to 8x8 for display\n",
                "    plt.imshow(img.reshape(8, 8), cmap='gray')\n",
                "    plt.title(f\"Pred: {predicted_label} ({confidence:.1f}%)\\nActual: {true_label}\")\n",
                "    plt.axis('off')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (Pyodide)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
};
