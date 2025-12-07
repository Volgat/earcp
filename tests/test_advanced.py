"""
Advanced tests for EARCP.
Verifies theoretical guarantees and advanced features.
"""

import sys
import os
import unittest
import numpy as np

# ensure earcp is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from earcp.core.ensemble_weighting import AdaptiveEnsembleWeighting
from earcp.core.coherence_metrics import CoherenceMetrics

class TestAdvancedFeatures(unittest.TestCase):
    
    def test_regret_bound_convergence(self):
        """
        Verify that EARCP weights converge to the best expert in a static setting.
        Theory: Regret is sub-linear -> Avg Loss converges to Min Loss.
        """
        n_experts = 3
        # Use higher sensitivity to speed up convergence test
        weighting = AdaptiveEnsembleWeighting(n_experts=n_experts, eta_s=5.0)
        
        # Scenario: Expert 0 is perfect (loss=0), others are random
        for t in range(50):
            # Performance scores (Higher is better for EARCP)
            p = np.array([1.0, 0.2, 0.3])
            c = np.ones(3) 
            
            weighting.update_weights(p, c)
            
        final_weights = weighting.get_weights()
        
        # Expert 0 should dominate
        self.assertEqual(np.argmax(final_weights), 0)
        self.assertGreater(final_weights[0], 0.8, f"Weight for best expert should be high, got {final_weights[0]}")

    def test_adaptive_beta(self):
        """
        Verify that beta adapts when loss is increasing.
        """
        weighting = AdaptiveEnsembleWeighting(n_experts=2, adapt_beta=True, adaptation_rate=0.1)
        initial_beta = weighting.beta
        
        # Simulate increasing loss
        losses = np.linspace(0.1, 0.9, 20)
        
        for loss in losses:
            p = np.array([0.5, 0.5])
            c = np.array([0.5, 0.5])
            weighting.update_weights(p, c, ensemble_loss=loss)
            
        self.assertGreater(weighting.beta, initial_beta, f"Beta should increase. Started {initial_beta}, ended {weighting.beta}")

    def test_heterogeneous_coherence(self):
        """
        Verify that CoherenceMetrics can handle custom objects (strings).
        """
        def str_agreement(a, b):
            # Safe comparison even if numpy arrays
            val_a = a.item() if isinstance(a, np.ndarray) and a.size == 1 else a
            val_b = b.item() if isinstance(b, np.ndarray) and b.size == 1 else b
            return 1.0 if str(val_a)[0] == str(val_b)[0] else 0.0
            
        coherence = CoherenceMetrics(n_experts=2, agreement_fn=str_agreement)
        
        # Inputs: Wrapped strings
        pred1 = np.array(["Cat"])
        pred2 = np.array(["Cat"])
        pred3 = np.array(["Dog"])
        
        c_same = coherence.update([pred1, pred2])
        self.assertEqual(c_same[0], 1.0)
        
        c_diff = coherence.update([pred1, pred3])
        # Should be lower than 1.0
        self.assertLess(c_diff[0], 1.0)

if __name__ == '__main__':
    unittest.main()
