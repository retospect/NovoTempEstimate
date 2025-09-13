#!/usr/bin/env python3
"""
Evaluation script for protein temperature prediction model.

This script evaluates a trained model on test data and provides
comprehensive metrics and visualizations.

Usage:
    python scripts/evaluate_model.py --model-path models/best_model.pth [options]
"""

import argparse
import sys
from pathlib import Path
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from typing import List, Tuple, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from novotempestimate.model import load_model, ProteinTemperatureLSTM
from novotempestimate.trainer import ProteinDataset, collate_fn
from novotempestimate.data_reader import TemStaProReader
from novotempestimate.peptide_encoder import PeptideEncoder
from torch.utils.data import DataLoader


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate protein temperature prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the trained model checkpoint")
    parser.add_argument("--data-pattern", type=str, default="*testing*",
                       help="Pattern to match test data files")
    parser.add_argument("--output-dir", type=str, default="evaluation",
                       help="Output directory for evaluation results")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--max-length", type=int, default=1000,
                       help="Maximum sequence length")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Sample size for testing")
    parser.add_argument("--save-predictions", action="store_true",
                       help="Save predictions to CSV file")
    parser.add_argument("--create-plots", action="store_true",
                       help="Create evaluation plots")
    
    return parser.parse_args()


class ModelEvaluator:
    """Evaluator for protein temperature prediction model."""
    
    def __init__(self, model_path: str, output_dir: str = "evaluation"):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model
            output_dir: Output directory for results
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.encoder = None
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        print(f"Loading model from {self.model_path}")
        
        # Load checkpoint to get model config
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            vocab_size = config['vocab_size']
        else:
            # Fallback: assume standard vocab size
            self.encoder = PeptideEncoder(include_modifications=True)
            vocab_size = self.encoder.vocab_size
        
        # Create and load model
        self.model = ProteinTemperatureLSTM(
            vocab_size=vocab_size,
            device=self.device
        )
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        # Initialize encoder if not done
        if self.encoder is None:
            self.encoder = PeptideEncoder(include_modifications=True)
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Model info: {self.model.get_model_info()}")
    
    def load_test_data(self, data_pattern: str, max_length: int, 
                      batch_size: int, sample_size: int = None) -> DataLoader:
        """Load test data."""
        print(f"Loading test data with pattern: {data_pattern}")
        
        records = []
        with TemStaProReader() as reader:
            for record in reader.read_dataset(data_pattern):
                if record.temperature > 0 and len(record.sequence) > 10:
                    records.append(record)
                
                if sample_size and len(records) >= sample_size:
                    break
        
        print(f"Loaded {len(records)} test records")
        
        # Create dataset and dataloader
        dataset = ProteinDataset(records, self.encoder, max_length)
        dataloader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=0
        )
        
        return dataloader, records
    
    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on the test data."""
        print("Making predictions...")
        
        all_predictions = []
        all_targets = []
        
        self.model.eval()
        with torch.no_grad():
            for sequences, temperatures, lengths in dataloader:
                sequences = sequences.to(self.device)
                lengths = lengths.to(self.device)
                
                predictions = self.model(sequences, lengths)
                
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(temperatures.numpy())
        
        return np.array(all_predictions), np.array(all_targets)
    
    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {
            'mae': mean_absolute_error(targets, predictions),
            'mse': mean_squared_error(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions)),
            'r2': r2_score(targets, predictions),
            'pearson_r': pearsonr(targets, predictions)[0],
            'spearman_r': spearmanr(targets, predictions)[0],
            'mean_target': np.mean(targets),
            'std_target': np.std(targets),
            'mean_prediction': np.mean(predictions),
            'std_prediction': np.std(predictions),
            'min_target': np.min(targets),
            'max_target': np.max(targets),
            'min_prediction': np.min(predictions),
            'max_prediction': np.max(predictions)
        }
        
        # Calculate temperature range accuracy
        temp_ranges = [(0, 30), (30, 50), (50, 70), (70, 100)]
        for temp_min, temp_max in temp_ranges:
            mask = (targets >= temp_min) & (targets < temp_max)
            if np.sum(mask) > 0:
                range_mae = mean_absolute_error(targets[mask], predictions[mask])
                metrics[f'mae_{temp_min}_{temp_max}C'] = range_mae
        
        return metrics
    
    def create_plots(self, predictions: np.ndarray, targets: np.ndarray):
        """Create evaluation plots."""
        print("Creating evaluation plots...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Scatter plot: Predicted vs Actual
        ax1 = axes[0, 0]
        ax1.scatter(targets, predictions, alpha=0.6, s=20)
        
        # Add perfect prediction line
        min_temp = min(np.min(targets), np.min(predictions))
        max_temp = max(np.max(targets), np.max(predictions))
        ax1.plot([min_temp, max_temp], [min_temp, max_temp], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Temperature (°C)')
        ax1.set_ylabel('Predicted Temperature (°C)')
        ax1.set_title('Predicted vs Actual Temperature')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add R² to the plot
        r2 = r2_score(targets, predictions)
        ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 2. Residual plot
        ax2 = axes[0, 1]
        residuals = predictions - targets
        ax2.scatter(targets, residuals, alpha=0.6, s=20)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Actual Temperature (°C)')
        ax2.set_ylabel('Residuals (Predicted - Actual)')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribution of residuals
        ax3 = axes[1, 0]
        ax3.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('Residuals (°C)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Residuals')
        ax3.grid(True, alpha=0.3)
        
        # Add statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        ax3.text(0.05, 0.95, f'Mean: {mean_residual:.2f}°C\nStd: {std_residual:.2f}°C', 
                transform=ax3.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 4. Temperature range analysis
        ax4 = axes[1, 1]
        temp_ranges = [(0, 30), (30, 50), (50, 70), (70, 100)]
        range_names = ['0-30°C', '30-50°C', '50-70°C', '70-100°C']
        range_maes = []
        range_counts = []
        
        for temp_min, temp_max in temp_ranges:
            mask = (targets >= temp_min) & (targets < temp_max)
            count = np.sum(mask)
            range_counts.append(count)
            
            if count > 0:
                mae = mean_absolute_error(targets[mask], predictions[mask])
                range_maes.append(mae)
            else:
                range_maes.append(0)
        
        bars = ax4.bar(range_names, range_maes, alpha=0.7)
        ax4.set_xlabel('Temperature Range')
        ax4.set_ylabel('Mean Absolute Error (°C)')
        ax4.set_title('MAE by Temperature Range')
        ax4.grid(True, alpha=0.3)
        
        # Add count labels on bars
        for bar, count in zip(bars, range_counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'n={count}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "evaluation_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {plot_path}")
        
        plt.show()
    
    def evaluate(self, data_pattern: str, max_length: int, batch_size: int,
                sample_size: int = None, save_predictions: bool = False,
                create_plots: bool = False) -> Dict[str, Any]:
        """
        Run complete evaluation.
        
        Args:
            data_pattern: Pattern to match test data files
            max_length: Maximum sequence length
            batch_size: Batch size for evaluation
            sample_size: Sample size for testing
            save_predictions: Whether to save predictions
            create_plots: Whether to create plots
            
        Returns:
            Evaluation results
        """
        # Load test data
        dataloader, records = self.load_test_data(
            data_pattern, max_length, batch_size, sample_size
        )
        
        # Make predictions
        predictions, targets = self.predict(dataloader)
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, targets)
        
        # Print metrics
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Number of test samples: {len(predictions)}")
        print(f"Mean Absolute Error: {metrics['mae']:.2f}°C")
        print(f"Root Mean Square Error: {metrics['rmse']:.2f}°C")
        print(f"R² Score: {metrics['r2']:.3f}")
        print(f"Pearson Correlation: {metrics['pearson_r']:.3f}")
        print(f"Spearman Correlation: {metrics['spearman_r']:.3f}")
        print("\nTemperature Statistics:")
        print(f"Target range: {metrics['min_target']:.1f}°C - {metrics['max_target']:.1f}°C")
        print(f"Prediction range: {metrics['min_prediction']:.1f}°C - {metrics['max_prediction']:.1f}°C")
        print("="*50)
        
        # Save detailed results
        results = {
            'metrics': metrics,
            'model_path': str(self.model_path),
            'num_samples': len(predictions),
            'data_pattern': data_pattern
        }
        
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")
        
        # Save predictions if requested
        if save_predictions:
            predictions_df = pd.DataFrame({
                'protein_id': [record.id for record in records],
                'sequence': [record.sequence for record in records],
                'organism': [record.organism for record in records],
                'actual_temperature': targets,
                'predicted_temperature': predictions,
                'absolute_error': np.abs(predictions - targets),
                'relative_error': np.abs(predictions - targets) / targets * 100
            })
            
            predictions_path = self.output_dir / "predictions.csv"
            predictions_df.to_csv(predictions_path, index=False)
            print(f"Predictions saved to {predictions_path}")
        
        # Create plots if requested
        if create_plots:
            self.create_plots(predictions, targets)
        
        return results


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model_path, args.output_dir)
    
    try:
        # Run evaluation
        results = evaluator.evaluate(
            data_pattern=args.data_pattern,
            max_length=args.max_length,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            save_predictions=args.save_predictions,
            create_plots=args.create_plots
        )
        
        print(f"\n✅ Evaluation completed! Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
