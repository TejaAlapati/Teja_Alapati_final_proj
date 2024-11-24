import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits # importing pre-defined dataset
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical

digits= load_digits()
dir(digits)
plt.gray()
for i in range(10):
    plt.matshow(digits.images[i])
    
X = digits.data
y = digits.target
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_categorical = to_categorical(y, num_classes=len(np.unique(y)))
kf = KFold(n_splits=3, shuffle=True, random_state=42)
def calc_metrics_multiclass(conf_matrix):
    metrics_list = []
    for cm in conf_matrix:  # Confusion matrix per class
        TP, FN = cm[0][0], cm[0][1]
        FP, TN = cm[1][0], cm[1][1]
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        FPR = FP / (TN + FP)
        FNR = FN / (TP + FN)
        Precision = TP / (TP + FP)
        F1_measure = 2 * TP / (2 * TP + FP + FN)
        Accuracy = (TP + TN) / (TP + FP + FN + TN)
        Error_rate = (FP + FN) / (TP + FP + FN + TN)
        BACC = (TPR + TNR) / 2
        TSS = TPR - FPR
        HSS = (
            2 * (TP * TN - FP * FN)
            / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
            if (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN) != 0
            else 0
        )
        metrics_list.append(
            [TP, TN, FP, FN, TPR, TNR, FPR, FNR, Precision, F1_measure, Accuracy, Error_rate, BACC, TSS, HSS]
        )
    return np.mean(metrics_list, axis=0)

def get_metrics_multiclass(model, X_train, X_test, y_train, y_test, LSTM_flag=False):
    if LSTM_flag:
        # Reshape for LSTM
        X_train = X_train.reshape(X_train.shape[0], 8, 8)
        X_test = X_test.reshape(X_test.shape[0], 8, 8)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # Train the LSTM model
        lstm_model = Sequential()
        lstm_model.add(LSTM(128, input_shape=(8, 8), activation='relu'))
        lstm_model.add(Dense(64, activation='relu'))
        lstm_model.add(Dense(10, activation='softmax'))
        lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Predictions
        y_pred = np.argmax(lstm_model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
    else:
        # Train classical model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_true = y_test

    # Compute confusion matrices per class
    conf_matrix = multilabel_confusion_matrix(y_true, y_pred)
    metrics = calc_metrics_multiclass(conf_matrix)
    return metrics
metric_labels = ["TP", "TN", "FP", "FN", "TPR", "TNR", "FPR", "FNR", "Precision", "F1_measure", "Accuracy", "Error_rate", "BACC", "TSS", "HSS"]

# Store results for each iteration
for fold, (train_index, test_index) in enumerate(kf.split(X_scaled), start=1):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Store metrics for this fold
    rf_metrics = get_metrics_multiclass(RandomForestClassifier(random_state=42, n_jobs=-1), X_train, X_test, y_train, y_test)
    lr_metrics = get_metrics_multiclass(LogisticRegression(max_iter=500, random_state=42, n_jobs=-1), X_train, X_test, y_train, y_test)
    lstm_metrics = get_metrics_multiclass(None, X_train, X_test, y_train, y_test, LSTM_flag=True)

   
    fold_results = pd.DataFrame({
        "Metric": metric_labels,
        "Random Forest": rf_metrics,
        "Logistic Regression": lr_metrics,
        "LSTM": lstm_metrics,
    })

    
    print(f"Fold {fold} Results")
    display(fold_results)
    print("\n" + "-" * 50 + "\n")


