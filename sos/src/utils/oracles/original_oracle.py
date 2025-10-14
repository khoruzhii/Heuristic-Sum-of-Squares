import time
from typing import List, Any, Dict
from .oracle_base import OracleBase
from data_generation.monomials.monomials import Polynomial

class OriginalOracle(OracleBase):
    def __init__(self, use_basis_extension: bool = False, basis_extension_params: Dict = None):
        super().__init__(use_basis_extension, basis_extension_params)

    def predict_basis(self, poly_tokens: List[Any], *args, **kwargs) -> Dict:
        # User must provide the original basis as 'original_basis' in kwargs
        original_basis = kwargs.get('original_basis')
        poly = kwargs.get('poly')
        if original_basis is None:
            raise ValueError("OriginalOracle requires the 'original_basis' argument (list of Monomials)")
        start_time = time.time()
        predicted_basis = list(original_basis)
        oracle_time = time.time() - start_time
        result = {
            'basis': predicted_basis,
            'time': oracle_time
        }
        if self.use_basis_extension:
            from utils.basis_extension import basis_extension
            if poly is None:
                raise ValueError("OriginalOracle with basis extension requires the 'poly' argument (Polynomial object)")
            ext_start = time.time()
            extended_basis = basis_extension(predicted_basis, poly, **self.basis_extension_params)
            ext_time = time.time() - ext_start
            result['basis'] = extended_basis
            result['basis_extension_time'] = ext_time
        return result 