import sys
sys.path.append('.')
import unittest
from unittest.mock import MagicMock
import numpy as np

from src.data.SCMGenerator import SCMGenerator

class TestDomainRandomization(unittest.TestCase):
    def test_sample_noise_distribution(self):
        print("\n--- Test sample_noise logic ---")
        gen = SCMGenerator()
        
        # Test Laplace
        with unittest.mock.patch('numpy.random.laplace') as mock_laplace:
            mock_laplace.return_value = np.zeros((10, 2))
            gen.sample_noise((10, 2), noise_type='laplace')
            mock_laplace.assert_called_once()
            print("Laplace called correct numpy function.")

    def test_pipeline_randomization(self):
        print("\n--- Test Pipeline Randomization ---")
        gen = SCMGenerator(num_nodes=5)
        
        # Monkeypatch sample_noise to track calls
        counts = {'normal': 0, 'laplace': 0, 'gumbel': 0, 'cauchy': 0, 'uniform': 0}
        
        original_sample_noise = gen.sample_noise
        
        def spy_sample_noise(size, noise_type='normal'):
            if noise_type in counts:
                counts[noise_type] += 1
            # Return dummy noise to let pipeline proceed
            return np.zeros(size)
            
        gen.sample_noise = spy_sample_noise
        
        # Run pipeline 20 times
        print("Running pipeline 20 times...")
        for i in range(20):
            gen.generate_pipeline(num_samples_base=10, num_samples_per_intervention=10, random_noise_type=True)
            
        print(f"Noise Type Counts: {counts}")
        
        # Verify diversity
        # It's random, but with 20 runs and 5 types, highly likely to see >1 type.
        types_seen = sum(1 for k in counts if counts[k] > 0)
        print(f"Types seen: {types_seen}/5")
        
        self.assertGreater(types_seen, 1, "Should see more than 1 noise type in 20 runs!")

if __name__ == '__main__':
    unittest.main()
