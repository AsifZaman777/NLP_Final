# Neural Language Model Implementation

## Overview

This project implements a deep learning-based language model using LSTM (Long Short-Term Memory) networks for next-word prediction and text generation.

## Files Created

1. **`neural_lm.ipynb`** - Standalone notebook with complete implementation
2. **`tokenization.ipynb`** - Extended with neural LM sections (cells added at the end)
3. **`requirements.txt`** - Updated with PyTorch and Gensim dependencies

## Model Architecture

### LSTM Language Model

- **Embedding Layer**: 128-dimensional word embeddings
- **LSTM Layers**: 2 stacked LSTM layers with 256 hidden units
- **Dropout**: 0.3 for regularization
- **Output Layer**: Fully connected layer mapping to vocabulary size
- **Total Parameters**: ~1.5M parameters (varies with vocabulary size)

### Model Flow

```
Input Tokens → Embedding → LSTM Layers → Dropout → Linear → Output Logits
```

## Key Features

### 1. **Data Preparation**

- Uses BPE (Byte Pair Encoding) tokenization for subword units
- Vocabulary built from SMS spam dataset
- Sequences created with sliding window approach (sequence length = 10)
- Special tokens: `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`

### 2. **Training**

- **Loss Function**: CrossEntropyLoss (ignores padding tokens)
- **Optimizer**: Adam with learning rate 0.001
- **Learning Rate Scheduler**: ReduceLROnPlateau (reduces LR when validation loss plateaus)
- **Gradient Clipping**: Max norm of 5.0 to prevent exploding gradients
- **Epochs**: 20 (configurable)
- **Batch Size**: 64

### 3. **Evaluation Metrics**

- **Loss**: Cross-entropy loss on validation set
- **Perplexity**: exp(loss) - measures how well the model predicts the next word
  - Lower perplexity = better model
  - Perplexity of 1 = perfect prediction
  - Perplexity of vocab_size = random guessing

### 4. **Text Generation**

- **Temperature Sampling**: Controls randomness
  - Low temp (0.5): More deterministic, safer outputs
  - High temp (1.5): More creative, diverse outputs
- **Top-k Sampling**: Samples from k most likely tokens (reduces unlikely outputs)
- **Seed Text**: Can start generation from custom prompts

### 5. **Next-Word Prediction**

- Given context, predicts top-k most likely next words with probabilities
- Useful for autocomplete and text suggestion applications

## Usage

### Running the Notebook

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Open the notebook**:

```bash
jupyter notebook neural_lm.ipynb
```

3. **Run all cells in sequence** - The notebook is self-contained and includes:
   - Data loading and preprocessing
   - BPE tokenizer training
   - Model definition and training
   - Evaluation and visualization
   - Text generation examples

### Expected Outputs

1. **Training Curves** (`lstm_lm_training_curves.png`):

   - Loss curves (training vs validation)
   - Perplexity curves (training vs validation)

2. **Model Weights** (`best_lstm_lm.pt`):

   - Best model saved based on validation loss

3. **Tokenizer** (`bpe_tokenizer.json`):

   - Trained BPE tokenizer for inference

4. **Generated Text Examples**:
   - Text generated from seed words
   - Random text generation
   - Temperature comparison
   - Next-word predictions

## Results Interpretation

### Training Performance

- Monitor loss curves: should decrease over epochs
- Check for overfitting: gap between train and validation loss
- Perplexity: Lower is better
  - < 50: Excellent
  - 50-100: Good
  - > 100: May need more training or hyperparameter tuning

### Generated Text Quality

- **Coherence**: Do generated sentences make sense?
- **Diversity**: With different temperatures, outputs should vary
- **Context Awareness**: Model should follow seed text context

## Hyperparameters

You can experiment with these in the notebook:

```python
# Model architecture
EMBEDDING_DIM = 128      # Size of word embeddings
HIDDEN_DIM = 256         # LSTM hidden state size
NUM_LAYERS = 2           # Number of LSTM layers
DROPOUT = 0.3            # Dropout rate

# Training
NUM_EPOCHS = 20          # Training epochs
BATCH_SIZE = 64          # Batch size
SEQ_LENGTH = 10          # Input sequence length
LEARNING_RATE = 0.001    # Initial learning rate

# Generation
temperature = 0.8        # Sampling temperature
top_k = 40              # Top-k sampling parameter
max_length = 20         # Max tokens to generate
```

## Advanced Usage

### Custom Text Generation

```python
# Generate text with custom seed
generated = generate_text(
    model,
    seed_text="your prompt here",
    token2idx=token2idx,
    idx2token=idx2token,
    max_length=20,
    temperature=0.8,
    top_k=40
)
print(generated)
```

### Next-Word Prediction

```python
# Predict next words
predictions = predict_next_word(
    model,
    context="your context",
    token2idx=token2idx,
    idx2token=idx2token,
    top_k=10
)

for word, prob in predictions:
    print(f"{word}: {prob:.4f}")
```

## Technical Details

### Dataset Processing

- **Source**: SMS Spam Collection dataset
- **Preprocessing**:
  - Lowercasing
  - Removing URLs, emails, numbers
  - Removing punctuation
  - Stopword removal
  - Lemmatization
- **Tokenization**: BPE with 5000 vocab size

### Model Training Details

- **Device**: Automatic GPU detection (falls back to CPU)
- **Reproducibility**: Random seeds set for consistent results
- **Memory Optimization**: Gradient accumulation not needed for this dataset size
- **Validation**: 20% of sequences held out for validation

### Limitations

- Trained on SMS dataset (short messages)
- May not generalize well to long-form text
- Generated text may contain repetitions
- Performance depends on vocabulary coverage

## Troubleshooting

### Out of Memory Error

- Reduce `BATCH_SIZE` (try 32 or 16)
- Reduce `HIDDEN_DIM` (try 128)
- Reduce `SEQ_LENGTH` (try 5)

### Poor Generation Quality

- Train for more epochs
- Increase model capacity (more layers/hidden units)
- Adjust temperature and top_k during generation
- Check if vocabulary size is appropriate

### Training Not Converging

- Lower learning rate
- Increase batch size
- Check for data preprocessing issues
- Ensure gradient clipping is working

## Extensions and Improvements

Possible enhancements:

1. **Attention Mechanism**: Add attention for better long-range dependencies
2. **Bidirectional LSTM**: For better context understanding
3. **Larger Dataset**: Train on more diverse text
4. **Transfer Learning**: Fine-tune pre-trained embeddings (GloVe, Word2Vec)
5. **Beam Search**: For more diverse text generation
6. **N-gram Language Model Comparison**: Compare with traditional n-gram models
7. **Transformer Architecture**: Implement Transformer-based LM for comparison

## References

- Hochreiter & Schmidhuber (1997): Long Short-Term Memory
- Bengio et al. (2003): A Neural Probabilistic Language Model
- Mikolov et al. (2010): Recurrent Neural Network Based Language Model
- PyTorch Documentation: https://pytorch.org/docs/

## License

Educational project - free to use and modify.

---

**Created**: December 2025
**Task**: Neural Language Model Implementation for NLP Course
**Status**: Complete ✓
