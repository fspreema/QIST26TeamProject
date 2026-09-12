"""Temporary in-process precision experiment; does not modify TensorOrder files."""
import runpy
import sys
import numpy as np
sys.path.insert(0, '/usr/src/app/TensorOrder/src')
from tensor_network.tensor_apis.numpy_apis import NumpyAPI

original_init = NumpyAPI.__init__
original_arg = NumpyAPI.add_argument
def extended_init(self):
    original_init(self)
    self._entry_type = np.longdouble
def extended_arg(self, key, value):
    if key == 'entry_type' and value == 'float64':
        self._entry_type = np.longdouble
    else:
        original_arg(self, key, value)
NumpyAPI.__init__ = extended_init
NumpyAPI.add_argument = extended_arg
print('Diagnostic dtype:', np.dtype(np.longdouble), 'max exponent:', np.finfo(np.longdouble).maxexp, file=sys.stderr)
sys.argv = ['tensororder.py', '--planner=factor-Flow', '--weights=cachet', '--timeout=30']
runpy.run_path('/usr/src/app/TensorOrder/src/tensororder.py', run_name='__main__')
