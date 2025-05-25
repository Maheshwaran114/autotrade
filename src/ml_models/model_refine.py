#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Review and Refinement Module.

This module provides tools for evaluating, refining, and improving the
performance of machine learning models used in the trading system.
It includes functionality for model validation, performance comparison,
feature importance analysis, and iterative model improvement.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from pathlib import Path
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score
)

# Configure logging
logger = logging.getLogger(__name__)


class ModelReviewer:
    """
    Class for evaluating and reviewing the performance of ML models.
    
    This reviewer calculates various performance metrics, generates visualizations,
    and provides recommendations for model improvements.
    """
    
    def __init__(
        self,
        models_dir: str = "models",
        results_dir: str = "reports/model_review",
    ):
        """
        Initialize the model reviewer.
        
        Args:
            models_dir: Directory where models are stored
            results_dir: Directory to save review results
        """
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize containers
        self.models = {}
        self.datasets = {}
    
    def load_model(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        model_type: str = "classifier"
    ) -> Any:
        """
        Load a trained ML model.
        
        Args:
            model_name: Name of the model
            model_path: Path to the model file (if None, construct from models_dir)
            model_type: Type of the model ('classifier' or 'regressor')
            
        Returns:
            Loaded model
        """
        if model_path is None:
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            
        try:
            logger.info(f"Loading model from {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Store model
            self.models[model_name] = {
                'model': model,
                'path': model_path,
                'type': model_type
            }
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def load_dataset(
        self,
        dataset_name: str,
        features_path: str,
        labels_path: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load a dataset for model evaluation.
        
        Args:
            dataset_name: Name of the dataset
            features_path: Path to the features file
            labels_path: Path to the labels file
            
        Returns:
            Tuple of features DataFrame and labels Series
        """
        try:
            logger.info(f"Loading dataset from {features_path} and {labels_path}")
            
            # Load features and labels
            if features_path.endswith('.pkl'):
                features = pd.read_pickle(features_path)
            elif features_path.endswith('.parquet'):
                features = pd.read_parquet(features_path)
            else:
                features = pd.read_csv(features_path)
                
            if labels_path.endswith('.pkl'):
                labels = pd.read_pickle(labels_path)
            elif labels_path.endswith('.parquet'):
                labels = pd.read_parquet(labels_path)
            else:
                labels = pd.read_csv(labels_path)
                
            # If labels is a DataFrame with one column, convert to Series
            if isinstance(labels, pd.DataFrame) and len(labels.columns) == 1:
                labels = labels.iloc[:, 0]
                
            # Store dataset
            self.datasets[dataset_name] = {
                'features': features,
                'labels': labels
            }
            
            return features, labels
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            return None, None
    
    def evaluate_model(
        self,
        model_name: str,
        dataset_name: str,
        generate_plots: bool = True
    ) -> Dict:
        """
        Evaluate a model's performance on a dataset.
        
        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset
            generate_plots: Whether to generate performance plots
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
            
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")
            
        model_info = self.models[model_name]
        model = model_info['model']
        model_type = model_info['type']
        
        dataset = self.datasets[dataset_name]
        X = dataset['features']
        y = dataset['labels']
        
        logger.info(f"Evaluating model {model_name} on dataset {dataset_name}")
        
        # Make predictions
        try:
            y_pred = model.predict(X)
            
            # For classifiers, get probability predictions if available
            y_prob = None
            if model_type == 'classifier' and hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X)
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return {'error': str(e)}
        
        # Calculate metrics based on model type
        metrics = {}
        if model_type == 'classifier':
            metrics = self._evaluate_classifier(y, y_pred, y_prob)
        else:
            metrics = self._evaluate_regressor(y, y_pred)
        
        # Generate plots if requested
        if generate_plots:
            plots_dir = os.path.join(self.results_dir, model_name, dataset_name)
            os.makedirs(plots_dir, exist_ok=True)
            
            if model_type == 'classifier':
                plot_files = self._generate_classifier_plots(y, y_pred, y_prob, plots_dir)
            else:
                plot_files = self._generate_regressor_plots(y, y_pred, plots_dir)
                
            metrics['plots'] = plot_files
        
        # Add model analysis
        metrics['feature_importance'] = self._analyze_feature_importance(model, X.columns)
        
        # Generate report
        report_path = self._generate_report(model_name, dataset_name, metrics)
        metrics['report'] = report_path
        
        return metrics
    
    def _evaluate_classifier(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Evaluate a classifier model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (if available)
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Check if binary or multiclass
        unique_classes = np.unique(y_true)
        is_binary = len(unique_classes) == 2
        
        if is_binary:
            # Binary classification metrics
            metrics['precision'] = precision_score(y_true, y_pred, average='binary')
            metrics['recall'] = recall_score(y_true, y_pred, average='binary')
            metrics['f1'] = f1_score(y_true, y_pred, average='binary')
            
            # ROC AUC if probabilities available
            if y_prob is not None and y_prob.shape[1] == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                metrics['avg_precision'] = average_precision_score(y_true, y_prob[:, 1])
        else:
            # Multiclass metrics
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
            
            # Per-class metrics
            class_metrics = {}
            for cls in unique_classes:
                cls_name = str(cls)
                y_true_binary = (y_true == cls).astype(int)
                y_pred_binary = (y_pred == cls).astype(int)
                
                class_metrics[cls_name] = {
                    'precision': precision_score(y_true_binary, y_pred_binary, average='binary'),
                    'recall': recall_score(y_true_binary, y_pred_binary, average='binary'),
                    'f1': f1_score(y_true_binary, y_pred_binary, average='binary')
                }
                
                # ROC AUC if probabilities available
                if y_prob is not None:
                    cls_idx = np.where(unique_classes == cls)[0][0]
                    class_metrics[cls_name]['roc_auc'] = roc_auc_score(
                        y_true_binary, y_prob[:, cls_idx]
                    )
            
            metrics['class_metrics'] = class_metrics
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        return metrics
    
    def _evaluate_regressor(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray
    ) -> Dict:
        """
        Evaluate a regressor model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        # Calculate regression metrics
        metrics['mse'] = np.mean((y_true - y_pred)**2)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        metrics['r2'] = 1 - metrics['mse'] / np.var(y_true) if np.var(y_true) > 0 else 0
        
        return metrics
    
    def _generate_classifier_plots(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        plots_dir: str
    ) -> Dict[str, str]:
        """
        Generate plots for a classifier model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            plots_dir: Directory to save plots
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        plot_files = {}
        
        # Confusion matrix heatmap
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_plot = os.path.join(plots_dir, "confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(cm_plot)
        plt.close()
        plot_files['confusion_matrix'] = cm_plot
        
        # Check if binary or multiclass
        unique_classes = np.unique(y_true)
        is_binary = len(unique_classes) == 2
        
        # ROC curve for binary classification or one-vs-rest for multiclass
        if y_prob is not None:
            if is_binary:
                # Binary ROC curve
                plt.figure(figsize=(10, 8))
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_score(y_true, y_prob[:, 1]):.3f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                roc_plot = os.path.join(plots_dir, "roc_curve.png")
                plt.tight_layout()
                plt.savefig(roc_plot)
                plt.close()
                plot_files['roc_curve'] = roc_plot
                
                # Precision-Recall curve
                plt.figure(figsize=(10, 8))
                precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
                plt.plot(recall, precision, label=f'PR (Avg Precision = {average_precision_score(y_true, y_prob[:, 1]):.3f})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend()
                pr_plot = os.path.join(plots_dir, "pr_curve.png")
                plt.tight_layout()
                plt.savefig(pr_plot)
                plt.close()
                plot_files['pr_curve'] = pr_plot
            else:
                # One-vs-Rest ROC curves for multiclass
                plt.figure(figsize=(10, 8))
                for i, cls in enumerate(unique_classes):
                    y_true_binary = (y_true == cls).astype(int)
                    y_score = y_prob[:, i]
                    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                    plt.plot(
                        fpr, tpr, 
                        label=f'Class {cls} (AUC = {roc_auc_score(y_true_binary, y_score):.3f})'
                    )
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curves (One-vs-Rest)')
                plt.legend()
                roc_plot = os.path.join(plots_dir, "roc_curves.png")
                plt.tight_layout()
                plt.savefig(roc_plot)
                plt.close()
                plot_files['roc_curves'] = roc_plot
        
        return plot_files
    
    def _generate_regressor_plots(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        plots_dir: str
    ) -> Dict[str, str]:
        """
        Generate plots for a regressor model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            plots_dir: Directory to save plots
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        plot_files = {}
        
        # Scatter plot of true vs predicted
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted Values')
        scatter_plot = os.path.join(plots_dir, "true_vs_predicted.png")
        plt.tight_layout()
        plt.savefig(scatter_plot)
        plt.close()
        plot_files['true_vs_predicted'] = scatter_plot
        
        # Residual plot
        plt.figure(figsize=(10, 8))
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='k', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        residual_plot = os.path.join(plots_dir, "residuals.png")
        plt.tight_layout()
        plt.savefig(residual_plot)
        plt.close()
        plot_files['residuals'] = residual_plot
        
        # Residual histogram
        plt.figure(figsize=(10, 8))
        plt.hist(residuals, bins=30)
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        hist_plot = os.path.join(plots_dir, "residual_histogram.png")
        plt.tight_layout()
        plt.savefig(hist_plot)
        plt.close()
        plot_files['residual_histogram'] = hist_plot
        
        return plot_files
    
    def _analyze_feature_importance(
        self,
        model: Any,
        feature_names: List[str]
    ) -> Dict:
        """
        Analyze feature importance of a model.
        
        Args:
            model: Trained model
            feature_names: Names of features
            
        Returns:
            Dictionary with feature importance analysis
        """
        importance_data = {}
        
        try:
            # Extract feature importance if available
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_)
                if importances.ndim > 1:
                    importances = importances.mean(axis=0)
            else:
                logger.warning("Model does not have feature_importances_ or coef_ attribute")
                return {'error': "Feature importance not available for this model type"}
            
            # Create DataFrame of feature importances
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # Convert to dict for JSON serialization
            importance_data = importance_df.to_dict(orient='records')
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
            importance_data = {'error': str(e)}
        
        return importance_data
    
    def _generate_report(
        self,
        model_name: str,
        dataset_name: str,
        metrics: Dict
    ) -> str:
        """
        Generate a HTML report for model evaluation.
        
        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset
            metrics: Dictionary with evaluation metrics
            
        Returns:
            Path to the generated report
        """
        # Create report directory
        report_dir = os.path.join(self.results_dir, model_name, dataset_name)
        os.makedirs(report_dir, exist_ok=True)
        
        # Report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(report_dir, f"evaluation_{timestamp}.html")
        
        # Build HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{model_name} - Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .bad {{ color: red; }}
                .plot {{ margin: 20px 0; text-align: center; }}
                .plot img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>{model_name} - Model Evaluation Report</h1>
            <p>Dataset: {dataset_name}</p>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Performance Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
        """
        
        # Add metrics to table
        for metric, value in metrics.items():
            # Skip complex or non-scalar metrics
            if metric in ['confusion_matrix', 'class_metrics', 'plots', 'feature_importance', 'report']:
                continue
                
            # Format value and determine CSS class
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
                
                # Add CSS class based on metric and value
                css_class = ""
                if metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'r2']:
                    if value >= 0.8:
                        css_class = "good"
                    elif value >= 0.6:
                        css_class = "warning"
                    else:
                        css_class = "bad"
            else:
                formatted_value = str(value)
                css_class = ""
            
            html += f"""
            <tr>
                <td>{metric.replace('_', ' ').title()}</td>
                <td class="{css_class}">{formatted_value}</td>
            </tr>
            """
        
        html += """
            </table>
        """
        
        # Add class metrics if available
        if 'class_metrics' in metrics:
            html += """
            <h2>Class-Specific Metrics</h2>
            <table>
                <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                </tr>
            """
            
            for cls, cls_metrics in metrics['class_metrics'].items():
                html += f"""
                <tr>
                    <td>{cls}</td>
                    <td class="{'good' if cls_metrics['precision'] >= 0.8 else ('warning' if cls_metrics['precision'] >= 0.6 else 'bad')}">
                        {cls_metrics['precision']:.4f}
                    </td>
                    <td class="{'good' if cls_metrics['recall'] >= 0.8 else ('warning' if cls_metrics['recall'] >= 0.6 else 'bad')}">
                        {cls_metrics['recall']:.4f}
                    </td>
                    <td class="{'good' if cls_metrics['f1'] >= 0.8 else ('warning' if cls_metrics['f1'] >= 0.6 else 'bad')}">
                        {cls_metrics['f1']:.4f}
                    </td>
                </tr>
                """
            
            html += """
            </table>
            """
        
        # Add confusion matrix if available
        if 'confusion_matrix' in metrics:
            html += """
            <h2>Confusion Matrix</h2>
            <table>
            """
            
            for row in metrics['confusion_matrix']:
                html += "<tr>"
                for cell in row:
                    html += f"<td>{cell}</td>"
                html += "</tr>"
                
            html += """
            </table>
            """
        
        # Add feature importance if available
        if 'feature_importance' in metrics and not isinstance(metrics['feature_importance'], dict):
            html += """
            <h2>Feature Importance</h2>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Importance</th>
                </tr>
            """
            
            # Sort by importance
            for item in metrics['feature_importance']:
                html += f"""
                <tr>
                    <td>{item['feature']}</td>
                    <td>{item['importance']:.4f}</td>
                </tr>
                """
                
            html += """
            </table>
            """
        
        # Add plots if available
        if 'plots' in metrics:
            html += """
            <h2>Performance Plots</h2>
            """
            
            for plot_name, plot_path in metrics['plots'].items():
                if os.path.exists(plot_path):
                    plot_title = plot_name.replace('_', ' ').title()
                    rel_path = os.path.relpath(plot_path, os.path.dirname(report_file))
                    html += f"""
                    <div class="plot">
                        <h3>{plot_title}</h3>
                        <img src="{rel_path}" alt="{plot_title}">
                    </div>
                    """
        
        # Add recommendations
        html += """
            <h2>Model Review Recommendations</h2>
            <ul>
        """
        
        # Generate recommendations based on metrics
        recommendations = self._generate_recommendations(metrics)
        for rec in recommendations:
            html += f"<li>{rec}</li>"
            
        html += """
            </ul>
            
        </body>
        </html>
        """
        
        # Write to file
        with open(report_file, 'w') as f:
            f.write(html)
            
        logger.info(f"Model evaluation report generated at {report_file}")
        return report_file
    
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """
        Generate recommendations based on model metrics.
        
        Args:
            metrics: Dictionary with evaluation metrics
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check for low accuracy
        if 'accuracy' in metrics and metrics['accuracy'] < 0.7:
            recommendations.append(
                "Consider feature engineering to improve model accuracy. "
                "The current accuracy is below 70%, which may not be sufficient for reliable predictions."
            )
            
        # Check for class imbalance
        if 'class_metrics' in metrics:
            class_f1s = [m['f1'] for m in metrics['class_metrics'].values()]
            if max(class_f1s) - min(class_f1s) > 0.2:
                recommendations.append(
                    "There appears to be class imbalance or difficulty predicting certain classes. "
                    "Consider using class weights or resampling techniques to improve performance for all classes."
                )
        
        # Check for overfitting
        if 'feature_importance' in metrics and not isinstance(metrics['feature_importance'], dict):
            importance_values = [item['importance'] for item in metrics['feature_importance']]
            if importance_values and importance_values[0] > 0.5:
                recommendations.append(
                    "One feature dominates the model's decisions. "
                    "Consider regularization techniques or feature selection to create a more robust model."
                )
        
        # Add general recommendations
        recommendations.append(
            "Consider ensemble methods like stacking or blending to improve model performance."
        )
        
        recommendations.append(
            "Review the feature importance analysis and consider removing or transforming low-importance features."
        )
        
        return recommendations


class ModelRefiner:
    """
    Class for refining and improving ML models.
    
    This refiner implements techniques to enhance model performance,
    such as hyperparameter tuning, feature selection, and ensemble methods.
    """
    
    def __init__(
        self,
        models_dir: str = "models",
        results_dir: str = "reports/model_refinement",
    ):
        """
        Initialize the model refiner.
        
        Args:
            models_dir: Directory where models are stored
            results_dir: Directory to save refinement results
        """
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize reviewer for evaluation
        self.reviewer = ModelReviewer(models_dir, results_dir)
    
    def refine_model(
        self,
        original_model_name: str,
        dataset_name: str,
        refinement_techniques: List[str] = None,
        output_model_name: Optional[str] = None
    ) -> Dict:
        """
        Refine a model using specified techniques.
        
        Args:
            original_model_name: Name of the original model
            dataset_name: Name of the dataset
            refinement_techniques: List of refinement techniques to apply
            output_model_name: Name for the refined model (if None, construct from original name)
            
        Returns:
            Dictionary with refinement results
        """
        # Default refinement techniques
        if refinement_techniques is None:
            refinement_techniques = ['hyperparameter_tuning', 'feature_selection']
            
        # Default output model name
        if output_model_name is None:
            output_model_name = f"{original_model_name}_refined"
            
        logger.info(f"Refining model {original_model_name} using techniques: {refinement_techniques}")
        
        # Load original model and dataset
        original_model = self.reviewer.load_model(original_model_name)
        X, y = self.reviewer.datasets.get(dataset_name, (None, None))
        
        if original_model is None or X is None or y is None:
            error_msg = f"Failed to load model {original_model_name} or dataset {dataset_name}"
            logger.error(error_msg)
            return {'error': error_msg}
        
        # Dictionary to store refinement results
        results = {
            'original_model': original_model_name,
            'refined_model': output_model_name,
            'techniques_applied': [],
            'performance_improvement': {}
        }
        
        # Get baseline performance
        baseline_metrics = self.reviewer.evaluate_model(
            original_model_name, dataset_name, generate_plots=False
        )
        
        # Apply refinement techniques
        refined_model = original_model
        
        for technique in refinement_techniques:
            try:
                if technique == 'hyperparameter_tuning':
                    refined_model = self._apply_hyperparameter_tuning(refined_model, X, y)
                    results['techniques_applied'].append('hyperparameter_tuning')
                    
                elif technique == 'feature_selection':
                    refined_model = self._apply_feature_selection(refined_model, X, y)
                    results['techniques_applied'].append('feature_selection')
                    
                elif technique == 'ensemble':
                    refined_model = self._apply_ensemble(refined_model, X, y)
                    results['techniques_applied'].append('ensemble')
                    
            except Exception as e:
                logger.error(f"Error applying technique {technique}: {e}")
        
        # Save refined model
        refined_model_path = os.path.join(self.models_dir, f"{output_model_name}.pkl")
        try:
            with open(refined_model_path, 'wb') as f:
                pickle.dump(refined_model, f)
            logger.info(f"Refined model saved to {refined_model_path}")
            
            # Load the refined model into the reviewer
            self.reviewer.load_model(
                output_model_name, 
                refined_model_path,
                model_type=self.reviewer.models[original_model_name]['type']
            )
            
        except Exception as e:
            logger.error(f"Error saving refined model: {e}")
            results['error'] = str(e)
        
        # Evaluate refined model
        refined_metrics = self.reviewer.evaluate_model(output_model_name, dataset_name)
        
        # Calculate performance improvement
        improvement = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            if metric in baseline_metrics and metric in refined_metrics:
                improvement[metric] = refined_metrics[metric] - baseline_metrics[metric]
                
        results['performance_improvement'] = improvement
        results['baseline_metrics'] = {k: v for k, v in baseline_metrics.items() 
                                     if k not in ['plots', 'report']}
        results['refined_metrics'] = {k: v for k, v in refined_metrics.items() 
                                    if k not in ['plots', 'report']}
        
        # Generate comparison report
        report_path = self._generate_comparison_report(
            original_model_name, output_model_name, dataset_name,
            baseline_metrics, refined_metrics, results
        )
        results['report'] = report_path
        
        return results
    
    def _apply_hyperparameter_tuning(self, model, X, y):
        """
        Apply hyperparameter tuning to improve model performance.
        
        Args:
            model: Original model
            X: Features
            y: Labels
            
        Returns:
            Tuned model
        """
        from sklearn.model_selection import GridSearchCV
        
        logger.info("Applying hyperparameter tuning")
        
        # Define hyperparameter grid based on model type
        param_grid = {}
        
        # Check model type and define appropriate grid
        if hasattr(model, 'n_estimators'):
            # Random Forest or Gradient Boosting
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif hasattr(model, 'C'):
            # SVM or Logistic Regression
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'] if hasattr(model, 'kernel') else None
            }
            # Remove None values
            param_grid = {k: v for k, v in param_grid.items() if v is not None}
        
        # If we have a valid param_grid, perform grid search
        if param_grid:
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                scoring='accuracy' if len(np.unique(y)) > 1 else 'r2',
                n_jobs=-1
            )
            grid_search.fit(X, y)
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            return grid_search.best_estimator_
        
        # If no valid grid was defined, return original model
        logger.warning("No hyperparameter tuning performed (unsupported model type)")
        return model
    
    def _apply_feature_selection(self, model, X, y):
        """
        Apply feature selection to improve model performance.
        
        Args:
            model: Original model
            X: Features
            y: Labels
            
        Returns:
            Model with selected features
        """
        from sklearn.feature_selection import SelectFromModel
        
        logger.info("Applying feature selection")
        
        # Check if model supports feature importance
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            # Create a feature selector with the original model
            selector = SelectFromModel(model, threshold='mean')
            selector.fit(X, y)
            
            # Get selected feature mask and feature names
            selected_features = selector.get_support()
            feature_names = X.columns[selected_features].tolist() if hasattr(X, 'columns') else None
            
            logger.info(f"Selected {sum(selected_features)}/{len(selected_features)} features")
            if feature_names:
                logger.info(f"Selected features: {feature_names}")
                
            # Train a new model on the selected features
            X_selected = selector.transform(X)
            model.fit(X_selected, y)
            
            # Return a model that includes the selector
            from sklearn.pipeline import Pipeline
            return Pipeline([
                ('selector', selector),
                ('model', model)
            ])
        
        # If the model doesn't support feature importance, return original model
        logger.warning("No feature selection performed (unsupported model type)")
        return model
    
    def _apply_ensemble(self, model, X, y):
        """
        Apply ensemble methods to improve model performance.
        
        Args:
            model: Original model
            X: Features
            y: Labels
            
        Returns:
            Ensemble model
        """
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        
        logger.info("Applying ensemble methods")
        
        # Check if classification or regression task
        is_classification = len(np.unique(y)) > 1
        
        if is_classification:
            # For classification
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            from sklearn.linear_model import LogisticRegression
            
            # Create base estimators
            estimators = [
                ('original', model),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('svm', SVC(probability=True, random_state=42))
            ]
            
            # Create voting classifier
            ensemble_model = VotingClassifier(
                estimators=estimators,
                voting='soft'
            )
        else:
            # For regression
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import Ridge
            from sklearn.svm import SVR
            
            # Create base estimators
            estimators = [
                ('original', model),
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('ridge', Ridge(random_state=42))
            ]
            
            # Create voting regressor
            ensemble_model = VotingRegressor(estimators=estimators)
        
        # Fit ensemble model
        ensemble_model.fit(X, y)
        
        return ensemble_model
    
    def _generate_comparison_report(
        self,
        original_model_name: str,
        refined_model_name: str,
        dataset_name: str,
        baseline_metrics: Dict,
        refined_metrics: Dict,
        refinement_results: Dict
    ) -> str:
        """
        Generate a report comparing original and refined model performance.
        
        Args:
            original_model_name: Name of the original model
            refined_model_name: Name of the refined model
            dataset_name: Name of the dataset
            baseline_metrics: Metrics of the original model
            refined_metrics: Metrics of the refined model
            refinement_results: Results of refinement techniques
            
        Returns:
            Path to the generated report
        """
        # Create report directory
        report_dir = os.path.join(self.results_dir, f"{original_model_name}_to_{refined_model_name}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(report_dir, f"comparison_{timestamp}.html")
        
        # Build HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Refinement Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .improved {{ color: green; }}
                .unchanged {{ color: black; }}
                .degraded {{ color: red; }}
                .plot {{ margin: 20px 0; text-align: center; }}
                .plot img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Model Refinement Comparison Report</h1>
            <p>Original Model: {original_model_name}</p>
            <p>Refined Model: {refined_model_name}</p>
            <p>Dataset: {dataset_name}</p>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Refinement Techniques Applied</h2>
            <ul>
        """
        
        # Add refinement techniques
        for technique in refinement_results['techniques_applied']:
            html += f"<li>{technique}</li>"
        
        html += """
            </ul>
            
            <h2>Performance Comparison</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Original Model</th>
                    <th>Refined Model</th>
                    <th>Improvement</th>
                </tr>
        """
        
        # Add metrics comparison
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mse', 'rmse', 'r2']
        for metric in metrics_to_compare:
            if metric in baseline_metrics and metric in refined_metrics:
                original_value = baseline_metrics[metric]
                refined_value = refined_metrics[metric]
                improvement = refined_value - original_value
                
                # Determine if improvement is good or bad (for different metrics)
                if metric in ['mse', 'rmse']:
                    is_improved = improvement < 0  # Lower is better
                    improvement = -improvement  # Make positive for better display
                else:
                    is_improved = improvement > 0  # Higher is better
                
                # Determine CSS class
                if abs(improvement) < 0.001:
                    css_class = "unchanged"
                elif is_improved:
                    css_class = "improved"
                else:
                    css_class = "degraded"
                
                # Format values
                original_formatted = f"{original_value:.4f}"
                refined_formatted = f"{refined_value:.4f}"
                improvement_formatted = f"{improvement:.4f}" + (" (+" if is_improved else " (")
                if metric in ['mse', 'rmse']:
                    improvement_formatted += f"{abs(improvement/original_value)*100:.1f}%)"
                else:
                    improvement_formatted += f"{improvement/original_value*100:.1f}%)"
                
                html += f"""
                <tr>
                    <td>{metric.replace('_', ' ').upper()}</td>
                    <td>{original_formatted}</td>
                    <td>{refined_formatted}</td>
                    <td class="{css_class}">{improvement_formatted}</td>
                </tr>
                """
        
        html += """
            </table>
            
            <h2>Feature Importance Comparison</h2>
            <div style="display: flex; justify-content: space-between;">
                <div style="width: 48%;">
                    <h3>Original Model</h3>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Importance</th>
                        </tr>
        """
        
        # Add original feature importance
        if ('feature_importance' in baseline_metrics and 
                not isinstance(baseline_metrics['feature_importance'], dict)):
            for item in baseline_metrics['feature_importance']:
                html += f"""
                <tr>
                    <td>{item['feature']}</td>
                    <td>{item['importance']:.4f}</td>
                </tr>
                """
        
        html += """
                    </table>
                </div>
                <div style="width: 48%;">
                    <h3>Refined Model</h3>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Importance</th>
                        </tr>
        """
        
        # Add refined feature importance
        if ('feature_importance' in refined_metrics and 
                not isinstance(refined_metrics['feature_importance'], dict)):
            for item in refined_metrics['feature_importance']:
                html += f"""
                <tr>
                    <td>{item['feature']}</td>
                    <td>{item['importance']:.4f}</td>
                </tr>
                """
        
        html += """
                    </table>
                </div>
            </div>
            
            <h2>Conclusions</h2>
            <ul>
        """
        
        # Generate conclusions
        performance_changes = refinement_results.get('performance_improvement', {})
        key_metrics = ['accuracy', 'f1', 'roc_auc', 'r2']
        improved_metrics = [m for m in performance_changes if performance_changes.get(m, 0) > 0.01]
        degraded_metrics = [m for m in performance_changes if performance_changes.get(m, 0) < -0.01]
        
        # Overall assessment
        if any(m in improved_metrics for m in key_metrics) and not degraded_metrics:
            html += "<li>The refined model shows significant improvement across key metrics.</li>"
        elif improved_metrics and degraded_metrics:
            html += "<li>The refined model shows mixed results with improvements in some areas but degradation in others.</li>"
        elif not improved_metrics and degraded_metrics:
            html += "<li>The refinement techniques did not improve model performance. Consider trying different approaches.</li>"
        else:
            html += "<li>The refinement had minimal impact on model performance.</li>"
        
        # Specific improvements
        for metric in improved_metrics:
            improvement = performance_changes.get(metric, 0)
            if improvement > 0.05:
                html += f"<li>Significant improvement in {metric.replace('_', ' ').upper()} by {improvement:.3f} points.</li>"
        
        # Recommendations
        html += "<li>Recommendation: "
        if any(m in improved_metrics for m in key_metrics) and len(degraded_metrics) == 0:
            html += "Adopt the refined model for production use."
        elif len(improved_metrics) > len(degraded_metrics):
            html += "Consider using the refined model but monitor the metrics that showed degradation."
        else:
            html += "Continue using the original model while exploring other refinement techniques."
        html += "</li>"
        
        html += """
            </ul>
            
        </body>
        </html>
        """
        
        # Write to file
        with open(report_file, 'w') as f:
            f.write(html)
            
        logger.info(f"Comparison report generated at {report_file}")
        return report_file


# Simple command-line interface for testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Print usage information
    print("Model Review and Refinement Module")
    print("=================================")
    print("Usage examples:")
    print("1. python model_refine.py review day_classifier")
    print("2. python model_refine.py refine signal_filter")
    
    # Example code for using the module
    import sys
    
    if len(sys.argv) < 3:
        sys.exit(1)
        
    action = sys.argv[1]
    model_name = sys.argv[2]
    
    if action == 'review':
        # Review model
        reviewer = ModelReviewer()
        model = reviewer.load_model(model_name)
        
        # Mock dataset for testing
        import numpy as np
        import pandas as pd
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y_series = pd.Series(y)
        
        reviewer.datasets[model_name] = {'features': X_df, 'labels': y_series}
        
        # Evaluate model
        metrics = reviewer.evaluate_model(model_name, model_name)
        print(f"Evaluation report: {metrics.get('report')}")
        
    elif action == 'refine':
        # Refine model
        refiner = ModelRefiner()
        
        # Load model
        model = refiner.reviewer.load_model(model_name)
        
        # Mock dataset for testing
        import numpy as np
        import pandas as pd
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y_series = pd.Series(y)
        
        refiner.reviewer.datasets[model_name] = {'features': X_df, 'labels': y_series}
        
        # Refine model
        results = refiner.refine_model(model_name, model_name)
        print(f"Refinement results: {results.get('report')}")
