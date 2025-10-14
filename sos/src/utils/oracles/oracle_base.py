import time
from typing import List, Any, Dict

class OracleBase:
    def __init__(self, use_basis_extension: bool = False, basis_extension_params: Dict = None):
        self.use_basis_extension = use_basis_extension
        self.basis_extension_params = basis_extension_params or {}

    def predict_basis(self, poly_tokens: List[Any], *args, **kwargs) -> Dict:
        """
        Predict a basis for the given tokenized polynomial.
        Returns a dictionary with at least the following keys:
            - 'basis': the predicted basis (list of monomials or tokens)
            - 'time': time taken for the oracle call (float, seconds)
            - 'basis_extension_time': time taken for basis extension (float, seconds, optional)
        """
        raise NotImplementedError("Subclasses must implement predict_basis.") 