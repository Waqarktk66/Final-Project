
📄 Intrusion Detection System Using Machine Learning on UNSW-NB15 Dataset

Overview

This project focuses on evaluating various machine learning models for network intrusion detection 
using the UNSW-NB15 dataset. As cybersecurity threats continue to grow with the increasing number
of devices and network data, effective intrusion detection systems (IDS) are essential.
This work compares several popular ML models in terms of accuracy, precision, recall, F1-score, 
and computational efficiency to identify the most suitable approaches for different application scenarios.


Models Evaluated

The following machine learning models were trained and evaluated:

- Random Forest  
- Decision Tree  
- Multi-Layer Perceptron (MLP)  
- k-Nearest Neighbors (kNN)  
- Gradient Boosting Classifier  
- Logistic Regression  

Dataset

- Name: UNSW-NB15  
- Source: Australian Centre for Cyber Security (ACCS)  
- Details:  
  - 49 features  
  - 9 attack types  
  - 175,341 training records  
  - 82,332 testing records  
  - Mix of simulated attack traffic and real-world activities  

---

Key Findings

| Model                 | Accuracy (%) | Training Time (s) | Precision (%) | Recall (%) | F1-Score (%) |
|:----------------------|:-------------|:------------------|:---------------|:------------|:--------------|
| Random Forest        | 97.70       | 57.01               | 97.71            | 97.70        | 97.71           |
| Decision Tree          | 96.59         | Fast               | 96.59            | 96.59        | 96.59           |
| MLP (Neural Network)   | 96.42         | Moderate           | 96.42            | 96.42        | 96.42           |
| Gradient Boosting      | 95.73         | High               | 95.73            | 95.73        | 95.73           |
| kNN                    | 94.84         | 0.01 (train) / 6.36 (predict) | 94.84 | 94.84 | 94.84 |
| Logistic Regression    | 92.56         | 2.47               | 92.56            | 92.56        | 92.56           |

- Random Forest achieved the highest accuracy but required the longest training time.
- Decision Tree and MLP offered a good trade-off between accuracy and computational cost.
- kNN had fast training but slower prediction time, making it unsuitable for real-time systems.
- Logistic Regression was fastest overall but less accurate, making it suitable for lightweight applications.

---

Methodology

1. Dataset Preprocessing  
   - Feature selection  
   - Data cleaning and encoding  
   - Train-test split  

2. Model Training & Evaluation  
   - Train models on training set  
   - Evaluate on test set using accuracy, precision, recall, F1-score, training time, and prediction time  

3. Comparison & Analysis  
   - Identify best-performing models considering both detection performance and computational efficiency.


Conclusion

The study revealed that while Random Forest provides superior accuracy and balanced performance,
 it is computationally expensive. Decision Tree and MLP are recommended alternatives for scenarios where computational
 resources are limited or where real-time performance is necessary.
 The study also highlights the importance of balancing accuracy with system efficiency in intrusion detection systems.

Future Work

- Explore Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) for handling complex and evolving attack patterns.
- Apply advanced feature selection methods like PCA and Autoencoders.
- Develop hybrid models combining deep learning and machine learning techniques.
- Test model robustness against adversarial attacks.
- Optimize models for edge computing to improve real-time detection.
- Investigate semi-supervised and unsupervised learning for detecting zero-day and unknown attacks.


References

- Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). Military Communications and Information Systems Conference (MilCIS).
