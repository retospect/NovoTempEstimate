#!/usr/bin/env python3
"""
Inference script for protein temperature prediction.

This script provides a simple interface for predicting optimal growth
temperatures of proteins from their sequences using a trained model.

Usage:
    python scripts/predict_temperature.py --model-path models/best_model.pth --sequence "MKLLVL..."
    python scripts/predict_temperature.py --model-path models/best_model.pth --fasta-file sequences.fasta
"""

import argparse
import sys
from pathlib import Path
import torch
from Bio import SeqIO
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from novotempestimate.model import load_model
from novotempestimate.peptide_encoder import PeptideEncoder


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict protein optimal growth temperature",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--sequence", type=str, help="Single protein sequence to predict"
    )
    input_group.add_argument(
        "--fasta-file", type=str, help="FASTA file containing protein sequences"
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file for predictions (CSV format)",
    )
    parser.add_argument(
        "--max-length", type=int, default=1000, help="Maximum sequence length"
    )

    return parser.parse_args()


class TemperaturePredictor:
    """Simple interface for temperature prediction."""

    def __init__(self, model_path: str):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the trained model
        """
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.encoder = PeptideEncoder(include_modifications=True)

        # Load model
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        if "model_config" in checkpoint:
            config = checkpoint["model_config"]
            vocab_size = config["vocab_size"]
        else:
            vocab_size = self.encoder.vocab_size

        self.model = load_model(model_path, vocab_size, device=self.device)
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")

    def predict_single(self, sequence: str) -> float:
        """
        Predict temperature for a single sequence.

        Args:
            sequence: Protein sequence

        Returns:
            Predicted optimal growth temperature in Celsius
        """
        # Encode sequence
        try:
            encoded = self.encoder.encode_sequence(sequence)
            encoded = encoded.unsqueeze(0).to(self.device)  # Add batch dimension

            # Make prediction
            with torch.no_grad():
                prediction = self.model(encoded)
                temperature = prediction.item()

            return temperature

        except Exception as e:
            print(f"Error predicting temperature for sequence: {e}")
            return None

    def predict_batch(
        self, sequences: List[str], max_length: int = 1000
    ) -> List[float]:
        """
        Predict temperatures for multiple sequences.

        Args:
            sequences: List of protein sequences
            max_length: Maximum sequence length

        Returns:
            List of predicted temperatures
        """
        predictions = []

        for sequence in sequences:
            # Truncate if too long
            if len(sequence) > max_length:
                sequence = sequence[:max_length]

            temp = self.predict_single(sequence)
            predictions.append(temp)

        return predictions

    def predict_from_fasta(self, fasta_file: str, max_length: int = 1000) -> List[Dict]:
        """
        Predict temperatures from a FASTA file.

        Args:
            fasta_file: Path to FASTA file
            max_length: Maximum sequence length

        Returns:
            List of dictionaries with sequence info and predictions
        """
        results = []

        print(f"Reading sequences from {fasta_file}")

        try:
            for record in SeqIO.parse(fasta_file, "fasta"):
                sequence = str(record.seq)

                # Truncate if too long
                if len(sequence) > max_length:
                    sequence = sequence[:max_length]
                    truncated = True
                else:
                    truncated = False

                # Predict temperature
                temperature = self.predict_single(sequence)

                result = {
                    "id": record.id,
                    "description": record.description,
                    "sequence": sequence,
                    "length": len(sequence),
                    "truncated": truncated,
                    "predicted_temperature": temperature,
                }

                results.append(result)

                # Print progress
                if len(results) % 10 == 0:
                    print(f"Processed {len(results)} sequences...")

        except Exception as e:
            print(f"Error reading FASTA file: {e}")
            return []

        print(f"Processed {len(results)} sequences total")
        return results


def save_results_csv(results: List[Dict], output_file: str):
    """Save results to CSV file."""
    import pandas as pd

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def main():
    """Main prediction function."""
    args = parse_arguments()

    # Create predictor
    try:
        predictor = TemperaturePredictor(args.model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1

    # Make predictions
    if args.sequence:
        # Single sequence prediction
        print(f"\nPredicting temperature for sequence:")
        print(f"Length: {len(args.sequence)} amino acids")
        print(
            f"Sequence: {args.sequence[:50]}{'...' if len(args.sequence) > 50 else ''}"
        )

        temperature = predictor.predict_single(args.sequence)

        if temperature is not None:
            print(f"\n🌡️  Predicted optimal growth temperature: {temperature:.1f}°C")

            # Classify temperature range
            if temperature < 30:
                category = "Psychrophile (cold-loving)"
            elif temperature < 50:
                category = "Mesophile (moderate temperature)"
            elif temperature < 70:
                category = "Thermophile (heat-loving)"
            else:
                category = "Hyperthermophile (extreme heat-loving)"

            print(f"📊 Temperature category: {category}")
        else:
            print("❌ Failed to predict temperature")
            return 1

    elif args.fasta_file:
        # Multiple sequence prediction
        results = predictor.predict_from_fasta(args.fasta_file, args.max_length)

        if not results:
            print("❌ No sequences processed")
            return 1

        # Print summary
        temperatures = [
            r["predicted_temperature"]
            for r in results
            if r["predicted_temperature"] is not None
        ]

        if temperatures:
            print(f"\n📊 Prediction Summary:")
            print(f"Total sequences: {len(results)}")
            print(f"Successful predictions: {len(temperatures)}")
            print(f"Average temperature: {sum(temperatures)/len(temperatures):.1f}°C")
            print(
                f"Temperature range: {min(temperatures):.1f}°C - {max(temperatures):.1f}°C"
            )

            # Temperature distribution
            psychrophiles = sum(1 for t in temperatures if t < 30)
            mesophiles = sum(1 for t in temperatures if 30 <= t < 50)
            thermophiles = sum(1 for t in temperatures if 50 <= t < 70)
            hyperthermophiles = sum(1 for t in temperatures if t >= 70)

            print(f"\n🌡️  Temperature Distribution:")
            print(f"Psychrophiles (<30°C): {psychrophiles}")
            print(f"Mesophiles (30-50°C): {mesophiles}")
            print(f"Thermophiles (50-70°C): {thermophiles}")
            print(f"Hyperthermophiles (≥70°C): {hyperthermophiles}")

        # Save results if requested
        if args.output_file:
            save_results_csv(results, args.output_file)
        else:
            # Print first few results
            print(f"\n📋 First 5 predictions:")
            for i, result in enumerate(results[:5]):
                temp = result["predicted_temperature"]
                temp_str = f"{temp:.1f}°C" if temp is not None else "Failed"
                print(f"{i+1}. {result['id']}: {temp_str}")

            if len(results) > 5:
                print(f"... and {len(results) - 5} more")

    print("\n✅ Prediction completed!")
    return 0


if __name__ == "__main__":
    exit(main())
