"""
EARCP Robotics Example: Balancing Speed, Precision, and Safety.

This example simulates a robotic arm controller where three "experts" propose
motor commands. EARCP modulates their influence based on context.

Experts:
1. SpeedController: Aggressive acceleration, ignores obstacles.
2. PrecisionController: Slow, exact movements.
3. SafetyController: Conservative, obstacle-aware.

Scenario:
The robot moves towards a target. Halfway through, a virtual obstacle appears.
EARCP should detect the disagreement (Speed expert ignoring obstacle) and
shift weight to the Safety expert.
"""

import sys
import os
import numpy as np
import time

# Add parent directory to path to import earcp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from earcp.core.ensemble_weighting import AdaptiveEnsembleWeighting
from earcp.core.coherence_metrics import CoherenceMetrics

class MockRoboticsExpert:
    def __init__(self, name, style):
        self.name = name
        self.style = style  # 'aggressive', 'precise', 'safe'
        
    def predict(self, current_pos, target_pos, obstacle_dist):
        """
        Propose a velocity vector (1D for simplicity).
        Positive = move to target. Negative = retreat.
        """
        dist = target_pos - current_pos
        
        if self.style == 'aggressive':
            # Full speed ahead, ignore obstacles
            return min(dist, 10.0)
            
        elif self.style == 'precise':
            # Slow and steady
            return min(dist, 2.0)
            
        elif self.style == 'safe':
            # Stop if obstacle is close
            if obstacle_dist < 5.0:
                return 0.0
            return min(dist, 5.0)
        return 0.0

def run_simulation():
    # Setup
    experts = [
        MockRoboticsExpert("Speed", "aggressive"),
        MockRoboticsExpert("Precision", "precise"),
        MockRoboticsExpert("Safety", "safe")
    ]
    
    # Initialize EARCP components manually for this low-level simulation
    # (In a real app, you might use the high-level EARCP wrapper)
    weighting = AdaptiveEnsembleWeighting(n_experts=3, beta=0.6, eta_s=5.0)
    coherence = CoherenceMetrics(n_experts=3, mode='regression')
    
    current_pos = 0.0
    target_pos = 100.0
    obstacle_pos = 60.0 # Obstacle appears at pos 60
    
    print(f"{'Step':<5} | {'Pos':<6} | {'Obstacle':<8} | {'Speed':<6} {'Prec':<6} {'Safe':<6} | {'Weights (Speed/Prec/Safe)':<30} | {'Action':<6}")
    print("-" * 100)
    
    for t in range(20):
        # dynamic obstacle distance (infinite until we get close)
        dist_to_obs = obstacle_pos - current_pos if current_pos > 40 else 999.0
        
        # 1. Get Expert Predictions
        preds = np.array([exp.predict(current_pos, target_pos, dist_to_obs) for exp in experts])
        
        # 2. Compute Coherence (Agreement)
        # We need predictions as list of arrays for the metric class
        pred_arrays = [np.array([p]) for p in preds]
        c_scores = coherence.update(pred_arrays)
        
        # 3. Compute Performance (Mock ground truth)
        # In reality, this would be a collision check or path deviation check.
        # Here: Any command that moves INTO an obstacle (dist < 0) gets 0 performance.
        p_scores = np.ones(3)
        for i, p in enumerate(preds):
            if dist_to_obs < 1.0 and p > 0: # Collision!
                p_scores[i] = 0.0
            elif dist_to_obs < 5.0 and p > 2.0: # Unsafe speed near obstacle
                p_scores[i] = 0.5
        
        # 4. Update Weights
        weights = weighting.update_weights(p_scores, c_scores)
        
        # 5. Aggregate
        final_action = np.dot(weights, preds)
        
        # Move robot
        current_pos += final_action
        
        # Visualization
        w_str = f"[{weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f}]"
        preds_str = f"{preds[0]:<6.1f} {preds[1]:<6.1f} {preds[2]:<6.1f}"
        obs_str = f"{dist_to_obs:.1f}" if dist_to_obs < 100 else "None"
        
        print(f"{t:<5} | {current_pos:<6.1f} | {obs_str:<8} | {preds_str} | {w_str:<30} | {final_action:<6.2f}")

        if current_pos >= target_pos:
            print("Target Reached!")
            break
            
if __name__ == "__main__":
    run_simulation()
