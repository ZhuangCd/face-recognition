# Facial Emotion Recognition

A deep learning project for classifying facial expressions into seven emotion categories using Vision Transformers (ViT).

## About

This project was developed as part of the **Applied Machine Learning** course. The goal is to classify facial images into one of seven emotion categories:

- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😊 Happy
- 😢 Sad
- 😲 Surprise
- 😐 Neutral

## Approach

This project uses a **pretrained Vision Transformer (ViT) model** from [Hugging Face](https://huggingface.co/), specifically the [`abhilash88/face-emotion-detection`](https://huggingface.co/abhilash88/face-emotion-detection) model. The pretrained model was then **fine-tuned** on the FER-2013 Facial Emotion Recognition dataset to improve performance on our specific task.

### Why Transfer Learning?

Instead of training a model from scratch, we did transfer learning by:
1. Loading a pretrained ViT model already trained on facial emotion data
2. Fine-tuning it on our training dataset to adapt to the specific data distribution

This approach significantly reduces training time and typically yields better results, especially when working with limited data.

## Project Structure

```
├── main.ipynb              # Initial exploration and baseline predictions
├── fine_tune.ipynb         # Fine-tuning the pretrained model
├── fine_tune_predict.ipynb # Predictions using the fine-tuned model
├── finetuned/              # Saved fine-tuned model weights
├── train_labels.csv        # Training data labels
├── submission.csv          # Predictions output
└── sample_submission.csv   # Sample submission format
```

## Tech Stack

- **Python** 
- **PyTorch** - Deep learning framework
- **Hugging Face Transformers** - Pretrained models and utilities
- **Vision Transformer (ViT)** - Model architecture
- **Pandas** - Data manipulation
- **PIL/Pillow** - Image processing

## Getting Started

1. Install dependencies:
```bash
pip install torch torchvision transformers pandas pillow tqdm scikit-learn
```

2. Run the notebooks in order:
   - `main.ipynb` - Explore data and test baseline model
   - `fine_tune.ipynb` - Fine-tune the model on training data
   - `fine_tune_predict.ipynb` - Generate predictions with the fine-tuned model

## References

```bibtex
@misc{face-emotion-detection,
  author = {Abhilash},
  title = {ViT Face Emotion Detection},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {https://huggingface.co/abhilash88/face-emotion-detection}
}
```

## License

This project was created for educational purposes as part of a university course.
