from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def get_classifier(model_name="svm", random_state=42):
    """
    Returns an initialized Scikit-Learn classifier based on the requested model name.
    These models are designed to be trained on the flattened CNN embeddings 
    or the PCA/FDA reduced components.
    """
    model_name = model_name.lower()
    
    if model_name == "svm":
        # SVM is incredibly effective on high-dimensional CNN embeddings.
        # probability=True is needed if we want to display confidence scores in Streamlit.
        return SVC(kernel='rbf', probability=True, random_state=random_state)
        
    elif model_name == "logistic_regression":
        # This mimics your Lasso/Ridge assignment. It applies L2 (Ridge) penalty by default.
        # Excellent for pure linear explainability.
        return LogisticRegression(max_iter=1000, random_state=random_state)
        
    elif model_name == "random_forest":
        # Great out-of-the-box performance, doesn't require scaling, and gives feature importances.
        return RandomForestClassifier(n_estimators=100, random_state=random_state)
        
    elif model_name == "adaboost":
        # Similar to your A4 assignment! We use sklearn's highly optimized version here.
        return AdaBoostClassifier(n_estimators=100, random_state=random_state)
        
    elif model_name == "lda":
        # Linear Discriminant Analysis - relates directly to your A2 FDA assignment.
        return LinearDiscriminantAnalysis()
        
    else:
        raise ValueError(f"Classifier {model_name} is not supported.")

def train_classifier(model, X_train, y_train):
    """
    Trains the classifier on the extracted features.
    X_train should be shape (Samples, Features).
    """
    print(f"Training {model.__class__.__name__}...")
    model.fit(X_train, y_train)
    return model

def predict_classifier(model, X_test):
    """
    Returns both hard class predictions and probabilities (confidence scores).
    """
    predictions = model.predict(X_test)
    
    # Not all models support predict_proba natively, so we check first
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
    else:
        probabilities = None
        
    return predictions, probabilities
