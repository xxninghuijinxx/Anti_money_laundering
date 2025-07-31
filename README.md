# Anti-Money Laundering Detection

This project explores machine learning and deep learning approaches for detecting suspicious financial transactions that may indicate money laundering. It involves data preprocessing, dimensionality reduction, class balancing, and predictive modeling using both Random Forest and PyTorch-based neural networks.

---

## 📂 Project Structure

- `anti_money_laundering.ipynb`: Main notebook with end-to-end pipeline.
- `AML_sample.csv`: First 10 rows of the original dataset, provided as a sample.
- `.gitignore`: Excludes large files and temporary checkpoints.
- `README.md`: Project description and usage guidance.

> ⚠️ Note: The full dataset (`AML.csv`, ~470MB) is excluded due to GitHub file size restrictions.

---

## 📊 Dataset

The dataset used in this project is sourced from Kaggle.

**Original source:**  
[Anti-Money Laundering Dataset on Kaggle](https://www.kaggle.com)

The dataset used in this project was originally obtained from Kaggle, titled similarly to "Anti-Money Laundering" or "Suspicious Transaction Detection". The link is currently unavailable. Please note that this data is used strictly for non-commercial, educational purposes.

This dataset is used **only for educational and research purposes**. All rights belong to the original creators. If you use this project or dataset, please also cite the original Kaggle source.

---

## 🚀 Getting Started

1. Clone this repository:

   ```bash
   git clone git@github.com:xxninghuijinxx/Anti_money_laundering.git
   cd Anti_money_laundering
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn torch
   ```

3. Launch Jupyter Notebook and open:

   ```bash
   jupyter notebook
   ```

4. Run `anti_money_laundering.ipynb` to reproduce the results.

---

## 🧠 Machine Learning Pipeline

This project includes:

- Exploratory Data Analysis with pandas & seaborn
- Missing value handling
- One-hot encoding for categorical variables
- Feature scaling using `StandardScaler`
- Dimensionality reduction using `PCA`
- Undersampling with `RandomUnderSampler` to address class imbalance
- Classification using `RandomForestClassifier`
- Confusion matrix and accuracy evaluation

---

## 🔥 Deep Learning Module (PyTorch)

The neural network is built with PyTorch and includes:

- Custom model using `torch.nn.Module`
- MSE/CE loss and `Adam` optimizer
- Data batching via `TensorDataset` and `DataLoader`
- GPU-aware training (if CUDA is available)

---

## 🛠 Dependencies

| Library              | Use |
|----------------------|-----|
| pandas, numpy        | Data manipulation |
| matplotlib, seaborn  | Visualization |
| scikit-learn         | ML preprocessing & classification |
| imbalanced-learn     | Undersampling |
| torch (PyTorch)      | Neural network modeling |

---

## 🙋‍♂️ Author

**Ninghui Jin**   
📧 Email: xxninghuijinxx@gmail.com  
🌐 GitHub: [@xxninghuijinxx](https://github.com/xxninghuijinxx)

---

## 📄 License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

You are free to:

- ✅ Use, share, and modify this code for **non-commercial** purposes with attribution

You are **not permitted** to:

- ❌ Use this code or its derivatives for any **commercial purpose**

License: [https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/)  
© 2025 Ninghui Jin. All rights reserved.