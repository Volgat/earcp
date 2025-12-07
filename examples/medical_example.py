"""
EARCP Medical Example: Diagnostic Reliability & Hallucination Prevention.

This example simulates a diagnostic support system where three experts analyze a patient case.
EARCP assigns confidence scores based on consensus.

Experts:
1. SymptomExpert: Analyzes text reports (Simulated NLP).
2. LabExpert: Analyzes numerical blood work (Simulated Stats).
3. HistoryExpert: Checks patient history (Simulated Knowledge Graph).

Scenario:
- Case 1: Easy flu case. All experts agree. -> High Confidence.
- Case 2: Complex case. Symptoms suggest Viral, Lab suggests Bacterial. -> Low Coherence -> Low Confidence.
"""

import sys
import numpy as np
import os

# Add parent directory to path to import earcp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from earcp.core.ensemble_weighting import AdaptiveEnsembleWeighting
from earcp.core.coherence_metrics import CoherenceMetrics

class MockMedicalExpert:
    def __init__(self, name, expertise):
        self.name = name
        self.expertise = expertise
        
    def predict(self, case_id):
        # Simulated responses for different cases
        if case_id == 1: # Clear Case: Flu
            return "Influenza"
        
        elif case_id == 2: # Conflicting Case
            if self.expertise == 'symptoms':
                return "Viral Infection" # Symptoms look viral
            elif self.expertise == 'lab':
                return "Bacterial Infection" # White blood cells high
            elif self.expertise == 'history':
                return "Viral Infection" # Patient prone to viral
                
        return "Unknown"

def calculate_agreement(pred1, pred2):
    """Simple semantic agreement mock."""
    if pred1 == pred2:
        return 1.0
    if "Infection" in pred1 and "Infection" in pred2:
        return 0.5 # Partial agreement
    return 0.0

def run_simulation():
    experts = [
        MockMedicalExpert("NLP_Symptom", "symptoms"),
        MockMedicalExpert("Stat_Lab", "lab"),
        MockMedicalExpert("KG_History", "history")
    ]
    
    # Initialize EARCP
    weighting = AdaptiveEnsembleWeighting(n_experts=3)
    coherence = CoherenceMetrics(n_experts=3, agreement_fn=calculate_agreement, mode='classification')
    
    cases = [1, 2]
    
    print(f"{'Case':<5} | {'Exp1 (Symp)':<15} {'Exp2 (Lab)':<20} {'Exp3 (Hist)':<15} | {'Coherence':<10} | {'Status'}")
    print("-" * 100)
    
    for case in cases:
        # 1. Predict
        raw_preds = [exp.predict(case) for exp in experts]
        
        # 2. Compute Coherence
        # Convert strings to dummy arrays for the library (which usually expects arrays)
        # OR use the custom agreement_fn which handles the raw inputs if we bypass the standard class
        # For simplicity here, we manually compute coherence using the metric class structure
        
        # Hack: The library classes expect numpy arrays. Let's wrap strings in 1-element object arrays
        # This works because we passed a custom agreement_fn
        pred_arrays = [np.array([p]) for p in raw_preds]
        
        c_scores = coherence.update(pred_arrays)
        avg_coherence = np.mean(c_scores)
        
        # 3. Simulate performance (assuming we don't know the ground truth yet)
        # In deployment, performance is often estimated or T-1 feedback. 
        # Here we assume neutral performance to see Coherence effect.
        p_scores = np.array([0.8, 0.8, 0.8])
        
        # 4. Update Weights
        weights = weighting.update_weights(p_scores, c_scores)
        
        # Decision Logic
        top_expert_idx = np.argmax(weights)
        final_diagnosis = raw_preds[top_expert_idx]
        
        status = "[High Confidence]" if avg_coherence > 0.8 else "[Human Review Needed]"
        
        print(f"{case:<5} | {raw_preds[0]:<15} {raw_preds[1]:<20} {raw_preds[2]:<15} | {avg_coherence:<10.2f} | {status}")

if __name__ == "__main__":
    run_simulation()
