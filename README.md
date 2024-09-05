# Predictive Model Comparison: TensorFlow vs PyTorch

## Project Overview

This project implements and compares predictive models using two popular deep learning frameworks: TensorFlow and PyTorch. The main objective is to analyze the performance, ease of use, and characteristics of both frameworks in a practical scenario.

## Features

- Implementation of equivalent neural network models in TensorFlow and PyTorch
- Custom training loops with progress visualization using tqdm
- Performance comparison in terms of Mean Squared Error (MSE) and training speed
- Synthetic data generation for consistent testing
- Comprehensive error handling and debugging processes

## Requirements

- Python 3.7+
- TensorFlow 2.x
- PyTorch 1.x
- NumPy
- scikit-learn
- tqdm

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/James23xr/PredictiveModels.git
   cd PredictiveModels
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     .\venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script to train and compare the models:

```
python main.py
```

This script will:
1. Generate synthetic data
2. Train a model using TensorFlow
3. Train a model using PyTorch
4. Compare the performance of both models

## Project Structure

- `main.py`: The main script that orchestrates the entire process
- `requirements.txt`: List of Python dependencies
- `README.md`: This file, containing project documentation

## Results

The project compares the models based on:
- Mean Squared Error (MSE) on the test set
- Training speed (iterations per second)

In our tests:
- TensorFlow Model: MSE of 0.0129
- PyTorch Model: MSE of 0.0892
- TensorFlow trained at ~2.15 iterations per second
- PyTorch trained at ~511.84 iterations per second

Please note that results may vary depending on the hardware and specific versions of the libraries used.

## Challenges and Learnings

Throughout this project, several challenges were encountered and overcome:
- Resolving framework-specific errors, such as TensorFlow's AttributeError with loss functions
- Optimizing code to improve performance and reduce warnings
- Implementing comparable architectures and initialization methods in both frameworks
- Gaining insights into the strengths and characteristics of each framework

Key learnings include:
- Deep understanding of TensorFlow and PyTorch APIs
- Practical experience with custom training loops and gradient handling
- Insights into performance optimization in deep learning projects
- Enhanced debugging skills in machine learning contexts

## Future Improvements

- Implement more complex architectures (CNNs, RNNs)
- Explore advanced optimization techniques specific to each framework
- Investigate deployment strategies for production environments
- Expand the comparison to include more metrics and scenarios


Thank you for your interest in this project! Your feedback and contributions are greatly appreciated.
