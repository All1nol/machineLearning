# Breast Cancer Classification - Neural Network Enhancement

## Project Overview
Built a neural network to classify breast cancer using the Wisconsin Diagnostic Breast Cancer dataset.

## Original Model
- Architecture: 2 hidden layers with 4 nodes each
- Activation: ReLU (hidden), Sigmoid (output)
- Optimizer: SGD
- Batch Size: 50
- Epochs: 30
- Accuracy: ~65%

## Enhanced Model
- Architecture: 5 hidden layers (16 -> 32 -> 64 -> 32 -> 16 nodes)
- Activation: ReLU (hidden), Sigmoid (output)
- Optimizer: Adam
- Batch Size: 32
- Epochs: 50
- Accuracy: ~98.5%

## Changes Made

### 1. Increased Hidden Layers
- From 2 layers to 5 layers
- Allows network to learn more complex patterns

### 2. Increased Nodes Per Layer
- From 4 nodes to up to 64 nodes
- Pyramid architecture: 16 -> 32 -> 64 -> 32 -> 16
- More capacity to learn features

### 3. Changed Optimizer
- From SGD to Adam
- Better convergence for deeper networks
- Adaptive learning rate

### 4. Adjusted Training Parameters
- Batch size: 50 -> 32 (more frequent updates)
- Epochs: 30 -> 50 (more training cycles)

## Results

| Metric | Original | Enhanced |
|--------|----------|----------|
| Training Accuracy | 65.47% | 98.39% |
| Test Accuracy | 65.71% | 98.57% |
| Training Loss | 0.5721 | 0.0540 |
| Test Loss | 0.5729 | 0.0459 |
| Parameters | ~165 | 5,441 |

## Key Takeaways
- Deeper networks with more nodes significantly improve performance
- Adam optimizer outperforms SGD for this architecture
- Proper layer sizing (pyramid shape) helps with feature learning
- Enhanced model achieved ~33% improvement in accuracy
