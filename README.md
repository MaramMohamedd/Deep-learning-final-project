# Deep Learning Comparative Study: Time Series & Text Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

## 📌 Overview

This repository contains a systematic comparative study of deep learning architectures on **two fundamentally different problems**:

1. **Time Series Forecasting** (Jena Climate Dataset) - Predicting temperature
2. **Text Sentiment Analysis** (IMDB Reviews) - Binary classification

The goal is to understand **when and why** different architectures (RNN, LSTM, GRU, Attention RNN, Transformer) perform best.

## 🎯 Key Findings

| Problem | Best Model | Key Insight |
|---------|------------|--------------|
| Climate Forecasting | **LSTM / GRU** | Attention didn't help much due to short-term dependencies |
| Sentiment Analysis | **RNN + Attention** | Attention significantly improved long text understanding |

## 📂 Repository Structure
deep-learning-final-project/
├── climate_forecasting/ # Jena temperature prediction
│ └── notebooks/ # 5 model implementations
│ ├── simple_rnn.ipynb
│ ├── lstm.ipynb
│ ├── gru.ipynb
│ ├── attentionrnn.ipynb
│ └── temperature_fusion_transformer.ipynb
│
├── text_classification/ # IMDB sentiment analysis
│ └── notebooks/ # RNN + Attention
│ └── rnn_attention_text.ipynb
│
└── docs/ # Full project documentation + our amazing comprehensive presentation for transformers
└── project_documentation.md
└── project_documentation.md
|
│
└──figures/ # figures about loss curves , comparisons 
|
│
└── README.md
└── LICENSE


## 📊 Experiments & Results

### 1. Climate Forecasting (Jena Dataset)
- **Data**: 10 years of weather data (14 features, 420,551 samples)
- **Task**: Predict temperature 1 hour ahead
- **Models**: SimpleRNN, LSTM, GRU, Attention RNN, Transformer

**Performance Comparison** (Validation MAE):
| Model | MAE (°C) | Parameters |
|-------|----------|------------|
| SimpleRNN | 2.15 | 15K |
| LSTM | 1.48 | 45K |
| GRU | 1.51 | 38K |
| Attention RNN | 1.52 | 52K |
| Transformer | 1.67 | 85K |

**Conclusion**: LSTM/GRU capture temporal dependencies best; attention adds complexity without benefit for this short-horizon task.

### 2. Text Classification (IMDB)
- **Data**: 50,000 movie reviews (balanced: 25k positive, 25k negative)
- **Task**: Binary sentiment classification
- **Model**: RNN with Attention Mechanism

**Results**:
- Accuracy: 85.2%
- Precision: 0.84
- Recall: 0.86
- F1-Score: 0.85

**Conclusion**: Attention mechanism helps focus on sentiment-bearing words (e.g., "brilliant", "terrible").

## 🔬 Methodology

### For both tasks:
1. **Data preprocessing** (normalization, sequence padding, train/val/test split)
2. **Model architecture design** (consistent hyperparameters where possible)
3. **Training** (early stopping, learning rate scheduling)
4. **Evaluation** (task-specific metrics + visualization)
5. **Comparative analysis** (table + insights)

### Key differences handled:
- **Time series**: Timesteps = 120, Features = 14, Loss = MAE
- **Text**: Vocabulary = 10,000, Max length = 200, Loss = Binary crossentropy

## 💡 Lessons Learned

1. **Simple RNNs** work for very short sequences but suffer vanishing gradients
2. **LSTM/GRU** are robust defaults for time series (even with moderate sequence lengths)
3. **Attention shines with text** - interpretable + performance boost
4. **Transformer** needs careful tuning; not always better than RNNs for small data
5. **Always match architecture to problem** - no free lunch

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow/Keras** (RNN, LSTM, GRU, Attention layers)
- **Pandas/NumPy** (data processing)
- **Matplotlib/Seaborn** (visualization)
- **Scikit-learn** (metrics, preprocessing)
- **Jupyter Notebooks** (interactive development)

## 📝 Future Work

- [ ] Hyperparameter optimization (Optuna)
- [ ] Bidirectional RNNs for text task
- [ ] Multi-step forecasting for climate data
- [ ] Deploy best models with FastAPI
- [ ] Add SHAP/LIME explanations for text model

## 🤔 Why No Web App?

A Streamlit dashboard was partially implemented but not finalized due to time constraints. The focus remained on rigorous experimental comparison and clean notebook implementations. All code is functional and reproducible.

## 📚 References

- [Jena Climate Dataset](https://www.bgc-jena.mpg.de/wetter/) - Max Planck Institute
- [IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) - Maas et al. 2011
- "Long Short-Term Memory" - Hochreiter & Schmidhuber 1997
- "Attention is All You Need" - Vaswani et al. 2017

## 📧 Contact

- **Maram Mohamed** - [marammohamedhanafy@gmail.com] 
- **Nada Mohamed** - [nadamohamedabdrabo@gmail.com] 
- **Shahd walid** - [shahdelshafei18@gmail.com] 
- **Jana Wael** - [janawael1113@gmail.com] 
- **Rokaya Mostafa** - [rokayamostafamahrous@gmail.com] 
- **Alaa Mohamed Walid** - [alaa.m.walid88@gmail.com] 

## 📜 License

MIT License - feel free to use, modify, and share!