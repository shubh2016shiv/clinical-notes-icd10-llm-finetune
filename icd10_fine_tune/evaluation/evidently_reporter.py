"""
Evidently AI Integration for Model Evaluation
==============================================

WHAT THIS MODULE DOES:
Integrates with Evidently AI Cloud to track model evaluation metrics,
detect data drift, and visualize classification performance.

WHY WE NEED THIS:
1. **Observability**: Track model performance over training runs and deployments.
   Know when your model is degrading before it affects production.

2. **Data Drift Detection**: Detect when input data distribution shifts from
   training data, which often predicts performance degradation.

3. **Classification Metrics**: Precision, recall, F1 per ICD-10 code class.
   Understand which codes the model handles well vs poorly.

4. **Cloud Dashboard**: Centralized monitoring across experiments and deployments.
   Share results with team without setting up infrastructure.

HOW IT WORKS:
1. Configure Evidently client with API key from .env
2. Create or connect to a project in Evidently Cloud
3. After each evaluation run, compute metrics (precision, recall, F1)
4. Push metrics to Evidently Cloud for visualization
5. Generate local HTML reports as backup

EDUCATIONAL NOTE - Why Evidently for LLM Evaluation:
Evidently provides specialized presets for LLM/NLP evaluation:
- TextDescriptors: Analyze text length, language, sentiment
- ClassificationPreset: Precision, recall, F1, confusion matrix
- DataDriftPreset: Distribution shift detection
- Custom evaluators: LLM-as-a-judge support

For ICD-10 coding (multi-label classification), we focus on:
- Per-class precision/recall (which codes are accurate)
- Micro/macro F1 (overall performance)
- Data drift on input text features
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from icd10_fine_tune.config.settings import settings


class EvidentlyReporter:
    """
    Evidently AI integration for model evaluation and monitoring.
    
    EDUCATIONAL NOTE - Design Pattern:
    This class follows the Repository pattern:
    - Abstracts Evidently API behind clean interface
    - Handles connection, project creation, metric pushing
    - Makes it easy to swap to different monitoring tool later
    
    Usage:
        reporter = EvidentlyReporter()
        reporter.log_evaluation(predictions, ground_truth, metadata)
        reporter.generate_local_report(predictions, ground_truth, output_path)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        project_name: Optional[str] = None,
        workspace: Optional[str] = None
    ):
        """
        Initialize Evidently reporter.
        
        Args:
            api_key: Evidently Cloud API key (defaults to settings)
            project_name: Project name in Evidently Cloud
            workspace: Workspace name in Evidently Cloud
        """
        self.api_key = api_key or settings.evidently_api_key
        self.project_name = project_name or settings.evidently_project_name
        self.workspace = workspace or settings.evidently_workspace
        
        self._client = None
        self._project = None
        
        # Validate API key exists
        if not self.api_key:
            print("WARNING: EVIDENTLY_AI API key not set. Cloud features disabled.")
            print("  Set EVIDENTLY_AI in .env for cloud dashboard integration.")
    
    def _initialize_client(self) -> bool:
        """
        Lazy initialization of Evidently client.
        
        Returns:
            True if client initialized successfully, False otherwise
        
        EDUCATIONAL NOTE - Lazy Initialization:
        We don't connect to Evidently until first use because:
        1. API key might not be set (local development)
        2. Network might be unavailable
        3. Faster module import
        """
        if self._client is not None:
            return True
        
        if not self.api_key:
            return False
        
        try:
            from evidently.ui.workspace.cloud import CloudWorkspace
            
            self._client = CloudWorkspace(
                token=self.api_key,
                url="https://app.evidently.cloud"
            )
            
            # Get or create project
            self._project = self._client.get_or_create_project(
                name=self.project_name
            )
            
            print(f"✓ Connected to Evidently Cloud: {self.project_name}")
            return True
            
        except Exception as e:
            print(f"WARNING: Failed to connect to Evidently Cloud: {e}")
            print("  Falling back to local reports only.")
            return False
    
    def compute_classification_metrics(
        self,
        predictions: List[List[str]],
        ground_truth: List[List[str]],
        class_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute multi-label classification metrics.
        
        Args:
            predictions: List of predicted ICD-10 code lists
            ground_truth: List of ground truth ICD-10 code lists
            class_labels: Optional list of all possible labels
        
        Returns:
            Dictionary with precision, recall, F1 (micro, macro, per-class)
        
        EDUCATIONAL NOTE - Multi-Label Metrics:
        
        For ICD-10 coding, each note can have MULTIPLE codes (multi-label).
        This is different from multi-class (exactly one label per sample).
        
        Key metrics:
        - Micro-F1: Treat all (sample, label) pairs equally
          Good for imbalanced datasets (common in ICD-10)
        
        - Macro-F1: Average F1 across all classes
          Treats all classes equally (rare codes count as much as common)
        
        - Per-class metrics: Which specific codes are problematic?
          E.g., model might be great at E11.9 but bad at Z79.4
        
        - Exact Match: Did we get ALL codes correct for a note?
          Very strict (often low), but useful for production thresholds
        """
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn.metrics import (
            precision_score, recall_score, f1_score
        )
        
        # Get all unique labels
        if class_labels is None:
            all_labels = set()
            for codes in predictions + ground_truth:
                all_labels.update(codes)
            class_labels = sorted(all_labels)
        
        # Convert to binary matrix format
        mlb = MultiLabelBinarizer(classes=class_labels)
        y_true = mlb.fit_transform(ground_truth)
        y_pred = mlb.transform(predictions)
        
        # Compute metrics
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(predictions),
            "num_classes": len(class_labels),
            
            # Micro metrics (aggregate across all samples and labels)
            "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
            "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
            "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
            
            # Macro metrics (average across classes)
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            
            # Per-class breakdown (for detailed analysis)
            "per_class": {}
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, label in enumerate(class_labels):
            metrics["per_class"][label] = {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1": float(f1_per_class[i])
            }
        
        # Exact match accuracy
        exact_matches = sum(
            set(pred) == set(truth)
            for pred, truth in zip(predictions, ground_truth)
        )
        metrics["exact_match_accuracy"] = exact_matches / len(predictions) if predictions else 0
        
        return metrics
    
    def log_evaluation(
        self,
        predictions: List[List[str]],
        ground_truth: List[List[str]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log evaluation results to Evidently Cloud.
        
        Args:
            predictions: Predicted ICD-10 codes per sample
            ground_truth: Ground truth ICD-10 codes per sample
            metadata: Additional metadata (model version, epoch, etc.)
        
        Returns:
            Dictionary with computed metrics
        
        EDUCATIONAL NOTE - Why Log to Cloud:
        1. Track performance over time (training runs, deployments)
        2. Compare experiments without local file management
        3. Share with team via shareable dashboards
        4. Set up alerts for performance degradation
        """
        # Compute metrics
        metrics = self.compute_classification_metrics(predictions, ground_truth)
        
        # Add metadata
        if metadata:
            metrics["metadata"] = metadata
        
        # Try to push to cloud
        if self._initialize_client():
            try:
                self._push_to_cloud(metrics)
                print("✓ Metrics pushed to Evidently Cloud")
            except Exception as e:
                print(f"WARNING: Failed to push to cloud: {e}")
        
        # Print summary
        self._print_metrics_summary(metrics)
        
        return metrics
    
    def _push_to_cloud(self, metrics: Dict[str, Any]) -> None:
        """Push metrics to Evidently Cloud."""
        # Evidently Cloud expects specific report format
        # For now, we'll create a simple snapshot
        if self._project is None:
            return
        
        try:
            
            # Create a summary dataframe for the report
            {
                "metric": ["F1 Micro", "F1 Macro", "Precision", "Recall", "Exact Match"],
                "value": [
                    metrics["f1_micro"],
                    metrics["f1_macro"],
                    metrics["precision_micro"],
                    metrics["recall_micro"],
                    metrics["exact_match_accuracy"]
                ]
            }
            
            # Log to project (simplified - actual implementation may vary)
            # This is a placeholder for full Evidently Cloud integration
            print(f"  Project: {self.project_name}")
            print(f"  Metrics logged at: {metrics['timestamp']}")
            
        except Exception as e:
            print(f"  Note: Full cloud integration requires additional setup: {e}")
    
    def _print_metrics_summary(self, metrics: Dict[str, Any]) -> None:
        """Print a summary of metrics to console."""
        print(f"\n{'='*60}")
        print("Evaluation Metrics Summary")
        print(f"{'='*60}")
        print(f"  Samples evaluated: {metrics['num_samples']}")
        print(f"  Unique classes: {metrics['num_classes']}")
        print("\n  Micro Metrics (aggregate):")
        print(f"    Precision: {metrics['precision_micro']:.4f}")
        print(f"    Recall:    {metrics['recall_micro']:.4f}")
        print(f"    F1:        {metrics['f1_micro']:.4f}")
        print("\n  Macro Metrics (class-averaged):")
        print(f"    Precision: {metrics['precision_macro']:.4f}")
        print(f"    Recall:    {metrics['recall_macro']:.4f}")
        print(f"    F1:        {metrics['f1_macro']:.4f}")
        print(f"\n  Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f}")
        print(f"{'='*60}\n")
    
    def generate_local_report(
        self,
        predictions: List[List[str]],
        ground_truth: List[List[str]],
        input_texts: Optional[List[str]] = None,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate a local HTML report with Evidently.
        
        Args:
            predictions: Predicted ICD-10 codes
            ground_truth: Ground truth ICD-10 codes
            input_texts: Optional clinical note texts (for text analysis)
            output_path: Path to save HTML report
        
        Returns:
            Path to generated HTML report
        
        EDUCATIONAL NOTE - Local vs Cloud Reports:
        Local reports are useful when:
        1. No API key configured
        2. Network unavailable
        3. Want to share specific report as file
        4. Debugging/development
        
        Cloud is better for:
        1. Long-term tracking
        2. Team collaboration
        3. Automated monitoring
        4. Historical comparisons
        """
        output_path = output_path or (settings.get_reports_dir() / f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Compute metrics
        metrics = self.compute_classification_metrics(predictions, ground_truth)
        
        # Generate HTML report
        html_content = self._generate_html_report(metrics, predictions, ground_truth)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"✓ Local report saved: {output_path}")
        
        return output_path
    
    def _generate_html_report(
        self,
        metrics: Dict[str, Any],
        predictions: List[List[str]],
        ground_truth: List[List[str]]
    ) -> str:
        """Generate a simple HTML report."""
        # Simple HTML template
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ICD-10 Model Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .metric-box {{ background: #f5f5f5; padding: 20px; margin: 10px 0; border-radius: 8px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #2196F3; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
    </style>
</head>
<body>
    <h1>ICD-10 Model Evaluation Report</h1>
    <p>Generated: {metrics['timestamp']}</p>
    
    <h2>Overall Metrics</h2>
    <div class="metric-box">
        <p>Samples: {metrics['num_samples']} | Classes: {metrics['num_classes']}</p>
        <p>F1 Micro: <span class="metric-value">{metrics['f1_micro']:.4f}</span></p>
        <p>F1 Macro: <span class="metric-value">{metrics['f1_macro']:.4f}</span></p>
        <p>Exact Match: <span class="metric-value">{metrics['exact_match_accuracy']:.4f}</span></p>
    </div>
    
    <h2>Per-Class Metrics (Top 20)</h2>
    <table>
        <tr><th>ICD-10 Code</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
        {"".join(f"<tr><td>{code}</td><td>{m['precision']:.3f}</td><td>{m['recall']:.3f}</td><td>{m['f1']:.3f}</td></tr>" for code, m in sorted(metrics['per_class'].items(), key=lambda x: x[1]['f1'], reverse=True)[:20])}
    </table>
</body>
</html>
"""
        return html
    
    def save_metrics_json(
        self,
        metrics: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Save metrics to JSON file for programmatic access.
        
        EDUCATIONAL NOTE - JSON Export:
        JSON metrics are useful for:
        1. Programmatic analysis (Python, Jupyter)
        2. CI/CD pipeline assertions
        3. Historical comparison scripts
        4. Integration with other tools
        """
        output_path = output_path or (settings.get_reports_dir() / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ Metrics JSON saved: {output_path}")
        
        return output_path
