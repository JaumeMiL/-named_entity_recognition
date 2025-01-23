# Named Entity Recognition Project 🧠

**Exploring Neural Network Architectures for Advanced Entity Recognition**

## Overview 📑

This project focuses on Named Entity Recognition (NER), a critical task in natural language processing. Using the BILOU tagging scheme, the goal is to identify and classify relevant entities within user sentences. The project explores various machine learning architectures, including convolutional, recurrent, and transformer-based networks, to optimize performance on NER tasks.

## Key Features 🛠️

- **Data Preparation**:
  - Tokenization and padding of sequences for consistent input length.
  - One-hot encoding for entity labels to enable effective model training.
- **Neural Network Architectures**:
  - **Convolutional Networks (CNNs)**: Capture local patterns in text sequences.
  - **Recurrent Networks (RNNs)**: Use GRU and LSTM layers to model sequential dependencies.
  - **Transformers**: Leverage attention mechanisms for capturing global relationships.
- **Experimentation**:
  - Class balancing techniques to address imbalanced datasets.
  - Embedding size optimization for semantic understanding.
  - Regularization methods, including dropout, to mitigate overfitting.
- **Evaluation**:
  - Metrics include F1-score (macro), accuracy, and confusion matrices.
  - Visualization of training and validation curves to monitor performance.

## Project Structure 📂

1. **Data Exploration and Preprocessing**:
   - Analyzed the dataset to identify challenges like class imbalance.
   - Converted sentences to numerical sequences and padded them for uniformity.
2. **Architecture Design and Experimentation**:
   - Implemented and tested CNNs, RNNs (GRU, LSTM), and Transformers.
   - Compared configurations, including bidirectional layers and multi-head attention.
3. **Model Selection**:
   - Selected the Bidirectional GRU architecture with 128 units as the final model based on performance metrics and computational efficiency.
4. **Performance Evaluation**:
   - Assessed the final model on a separate test set for generalization.
   - Analyzed learning curves and generated normalized confusion matrices.

## Execution 🚀

1. Install Python and the necessary libraries (TensorFlow, Keras, Pandas, Matplotlib, etc.).
2. Run the notebook `P2_2_notebook_Marta_Jaume_Abril.ipynb` in a Jupyter environment.
3. Analyze outputs, including evaluation metrics, learning curves, and confusion matrices.

## Key Results 🔍

- **Best Model**: A Bidirectional GRU with 128 units achieved the highest F1-score and accuracy across validation and test datasets.
- **Embedding Size Optimization**: An embedding dimension of 1000 was identified as the optimal size, balancing semantic understanding and computational efficiency.
- **Regularization Impact**: Dropout was found to degrade model performance in this specific NER task, indicating the architecture's robustness without additional regularization.
- **Transformer Configurations**: While explored, Transformers did not significantly outperform recurrent architectures for this dataset, and their higher computational cost limited their utility.

## Conclusion 📝

This project highlights the effectiveness of a Bidirectional GRU architecture for Named Entity Recognition tasks. Through rigorous experimentation, the model demonstrates strong performance in recognizing and classifying entities, providing a reliable foundation for further enhancements, such as pre-trained embeddings or expanded datasets.

**Developed by**: Marta Juncarol Pi, Jaume Mora Ladària, Abril Risso Matas
