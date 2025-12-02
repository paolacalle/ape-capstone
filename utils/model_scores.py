
import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    roc_curve as sk_roc_curve,
    auc as sk_auc
)
from .seed_setter import SeedSetter
SeedSetter.set_seed() # from .env

class Scores:
    """ 
    Scores class for evaluating classification models.
    Implemented from scratch for self-learning reasons.
    """
    @staticmethod
    def valid_scoring_methods():
        return [
            "confusion_matrix",
            "prevalence",
            "accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "npv",
            "auc",
            "roc",
            "auc_scratch",
            "roc_scratch"
        ]
        
    @staticmethod
    def verify_scoring_methods(methods: List[str]):
        """
        Verify that all requested scoring methods are valid.
        Raises ValueError if any invalid methods are found.
        """
        valid = set(Scores.valid_scoring_methods())
        invalid = [m for m in methods if m not in valid]
        if invalid:
            raise ValueError(
                f"Invalid scoring methods: {invalid}. "
                f"Valid methods are: {sorted(valid)}"
            )
    
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """  
        Calculate the confusion matrix components: true positives, false positives, true negatives, 
        and false negatives.
        
        parameters:
            y_true: array-like of shape (n_samples,) - True labels
            y_pred: array-like of shape (n_samples,) - Predicted labels
            
        returns:
            tuple: (tp, fp, tn, fn) - True positives, false positives, true negatives, false negatives
        """
        
        # convert to numpy arrays if not already
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
            
        assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match"
        
        tp = np.sum((y_true == 1) & (y_pred == 1)) # true positives -- correctly predicted positive cases
        fp = np.sum((y_true == 0) & (y_pred == 1)) # false positives -- incorrectly predicted positive cases
        tn = np.sum((y_true == 0) & (y_pred == 0)) # true negatives -- correctly predicted negative cases
        fn = np.sum((y_true == 1) & (y_pred == 0)) # false negatives -- incorrectly predicted negative cases
        
        return tp, fp, tn, fn
    
    @staticmethod
    def prevalence(y_true):
        """ 
        Prevalence of the positive class: P(Y = 1).
        
        parameters:
            y_true: array-like of shape (n_samples,) - True labels
            
        returns:
            float: Prevalence of the positive class in the dataset
        """
        y_true = np.asarray(y_true)
        ap = np.sum(y_true == 1)
        total = len(y_true)
        return ap / total if total > 0 else np.nan
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """ 
        Calculate the accuracy of the predictions.
        Provides a measure of how often the classifier is correct.
        Answers: How often is the classifier correct?
        
        parameters:
            y_true: array-like of shape (n_samples,) - True labels
            y_pred: array-like of shape (n_samples,) - Predicted labels
            
        returns:
            float: Accuracy of the predictions
        """
        tp, fp, tn, fn = Scores.confusion_matrix(y_true, y_pred)
        total = tp + fp + tn + fn
        res = (tp + tn) / total if total > 0 else np.nan
        return res
    
    @staticmethod
    def sensitivity(y_true, y_pred):
        """
        Calculate the sensitivity (recall) of the predictions.
        Sensitivity measures the proportion of actual positives correctly identified.
        Answers: How often are actual positives correctly identified?
        
        parameters:
            y_true: array-like of shape (n_samples,) - True labels
            y_pred: array-like of shape (n_samples,) - Predicted labels
            
        returns:
            float: Sensitivity of the predictions
        """
        tp, _, _, fn = Scores.confusion_matrix(y_true, y_pred)
        ap = (tp + fn) # actual positives
        res = tp / ap if ap > 0 else np.nan
        return res
    
    @staticmethod
    def specificity(y_true, y_pred):
        """
        Calculate the specificity of the predictions.
        Specificity measures the proportion of actual negatives correctly identified.
        Answers: How often are actual negatives correctly identified?
        
        parameters:
            y_true: array-like of shape (n_samples,) - True labels
            y_pred: array-like of shape (n_samples,) - Predicted labels
            
        returns:
            float: Specificity of the predictions
        """
        _, fp, tn, _ = Scores.confusion_matrix(y_true, y_pred)
        an = (tn + fp) # actual negatives
        res = tn / an if an > 0 else np.nan
        return res
    
    @staticmethod
    def precision(y_true, y_pred):
        """
        Calculate the precision of the predictions.
        Precision measures the proportion of positive identifications that were actually correct.
        Answers: How often are positive predictions correct?
        
        parameters:
            y_true: array-like of shape (n_samples,) - True labels
            y_pred: array-like of shape (n_samples,) - Predicted labels
            
        returns:
            float: Precision of the predictions
        """
        tp, fp, _ , _ = Scores.confusion_matrix(y_true, y_pred)
        pp = tp + fp  # predicted positives
        res = tp / pp if pp > 0 else np.nan
        return res
    
    @staticmethod
    def npv(y_true, y_pred):
        """
        Calculate the negative predictive value (NPV) of the predictions.
        NPV measures the proportion of negative identifications that were actually correct.
        Answers: How often are negative predictions correct?
        
        parameters:
            y_true: array-like of shape (n_samples,) - True labels
            y_pred: array-like of shape (n_samples,) - Predicted labels
            
        returns:
            float: Negative predictive value of the predictions
        """
        _, _, tn, fn = Scores.confusion_matrix(y_true, y_pred)
        pn = (tn + fn)
        res = tn / pn if pn > 0 else np.nan
        return res
    
    @staticmethod
    def roc_scratch(y_true, y_pred_scores):
        """
        ROC needs: 
        - TPR at various thresholds
        - FPR at those same thresholds
        To compute these, we iterate through these possible thresholds; Uncovering 
        what the confusion matrix looks like at each threshold. Then, we calculate the 
        TPR and FPR for each threshold.
        
        parameters:
            y_true: array-like of shape (n_samples,) - True labels
            y_pred_scores: array-like of shape (n_samples,) - Predicted scores or probabilities
            
        returns:
            tuple: (fpr, tpr, thresholds) where each is an array of values
        
        """
        # we sort the thresholds in descending order because 
        # this allows us to move from the most confident positive predictions 
        # to the least confident
        thresholds = np.sort(np.unique(y_pred_scores))[::-1]
        
        tprs = []
        fprs = []

        for thresh in thresholds:
            # get the predicted labels based on the current threshold
            y_pred = (y_pred_scores >= thresh).astype(int)
            
            # calculate FPR = 1 - specificity = 1 - TNR
            fpr = 1 - Scores.specificity(y_true, y_pred)
            
            # calculate TPR = sensitivity = recall
            tpr = Scores.sensitivity(y_true, y_pred)
            
            # check for valid 
            tpr = tpr if tpr > 0 else 0.0
            fpr = fpr if fpr > 0 else 0.0
            
            tprs.append(tpr)
            fprs.append(fpr)
            
        # add (0,0) and (1,1) points to the ROC curve for completeness
        fprs = [0] + fprs + [1]
        tprs = [0] + tprs + [1]
        thresholds = np.r_[
            thresholds[0] + 1, # higher than max threshold
            thresholds, 
            thresholds[-1] - 1 # lower than min threshold
        ]
        return fprs, tprs, thresholds
    
    @staticmethod
    def roc(y_true, y_pred_scores):
        return sk_roc_curve(y_true, y_pred_scores)
    
    @staticmethod
    def auc_scratch(y_true, y_pred_scores):
        """
        AUC calculate the area under the ROC curve.
        
        parameters:
            y_true: array-like of shape (n_samples,) - True labels
            y_pred_scores: array-like of shape (n_samples,) - Predicted scores or probabilities
            
        returns:
            float: Area under the ROC curve
        """
        roc = Scores.roc(y_true, y_pred_scores)
        
        # Calculate the area under the ROC curve using the trapezoidal rule
        # which approximates the integral of the curve
        return np.trapezoid(roc[1], roc[0])
    
    @staticmethod
    def auc(y_true, y_pred_scores):
        res = Scores.roc(y_true, y_pred_scores)
        x = res[0]
        y = res[1]
        return sk_auc(x, y)
    
    @staticmethod
    def evaluate(
        y_true,
        y_pred,
        y_pred_scores,
        methods: list[str],
    ) -> Dict[str, float]:
        
        """
        Evaluate multiple scoring methods.
        """
        results = {}
        for method in methods:
            print(f"Evaluating method: {method}")
            if hasattr(Scores, method):
                func = getattr(Scores, method)
                if method in ["roc", "auc", "roc_scratch", "auc_scratch"]:
                    results[method] = func(y_true, y_pred_scores)
                elif method == "prevalence":
                    results[method] = func(y_true)  # doesn't need y_pred
                else:
                    results[method] = func(y_true, y_pred)
            else:
                raise ValueError(
                    f"Method {method} not found in Scores class. "
                    f"Valid methods include: {Scores.valid_scoring_methods()}"
                )
        return results