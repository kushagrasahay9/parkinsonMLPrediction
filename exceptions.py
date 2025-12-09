import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def find_exceptions(model, X_train, y_train):
    """
    Finds the training samples misclassified by the model.
    These are treated as 'exceptions'.
    """
    preds = model.predict(X_train)
    misclassified_idx = np.where(preds != y_train)[0]

    # Convert to numpy arrays to avoid index mismatch issues
    X_train_arr = np.array(X_train)
    y_train_arr = np.array(y_train)

    exceptions = X_train_arr[misclassified_idx]
    exception_labels = y_train_arr[misclassified_idx]

    print(f"[INFO] Found {len(exceptions)} exception samples.")
    return exceptions, exception_labels


def predict_with_exceptions(model, X_test, exceptions, exception_labels):
    """
    Predicts with exception correction.
    Uses 1-NN to check if a test sample is similar to an exception.
    Only overrides if the distance is very small.
    """
    preds = model.predict(X_test)
    if len(exceptions) == 0:
        return preds

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(exceptions, exception_labels)

    # Get both predicted label and distance to nearest exception
    distances, indices = knn.kneighbors(X_test)
    override_labels = exception_labels[indices.flatten()]

    # choose a small threshold – tweak if needed
    threshold = 0.3
    final_pred = preds.copy()
    for i in range(len(preds)):
        if distances[i][0] < threshold:
            final_pred[i] = override_labels[i]

    return final_pred
