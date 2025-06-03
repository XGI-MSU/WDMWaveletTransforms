"""run all the included unit tests"""

import pytest

if __name__ == '__main__':
    pytest.cmdline.main(['inverse_wavelet_test.py', 'forward_wavelet_test.py', 'run_harness_tests.py'])
