from typing import List, Dict, Optional, Tuple, Union
from .base import BaseProcessor, ProcessTarget
import torch
import itertools as it
import warnings
from dataclasses import dataclass
from enum import Enum
import itertools

@dataclass
class ProcessedMonomial:
    tokens: List[int]
    coefficient_value: Optional[float] = None

class MonomialProcessorPlus(BaseProcessor):
    """Processor that converts monomials to (coef_id, exponents) and optionally stores continuous coefficient values"""
    def __init__(self, num_variables: int, max_degree: int, max_coef: int, 
                 target: ProcessTarget = ProcessTarget.BOTH, 
                 rational_coefficients: List[Tuple[int, int]] = None,
                 continuous_coefficient: bool = False):
        super().__init__(target)
        self.num_variables = num_variables
        self.max_degree = max_degree
        self.max_coef = max_coef
        self.rational_coefficients = rational_coefficients
        self.continuous_coefficient = continuous_coefficient
        self.coef_to_id = self._create_coef_dict()
        self.id_to_coef = {v: k for k, v in self.coef_to_id.items()}
        self.special_to_id = self._create_special_dict()
        self.id_to_special = {v: k for k, v in self.special_to_id.items()}
        # Special ID for continuous coefficient token
        self.C_TOKEN_ID = (max(self.coef_to_id.values(), default=-1) + 1) if continuous_coefficient else None
            
    def _create_coef_dict(self) -> Dict[str, int]:
        """Map coefficients to IDs"""
        coef_dict = {}
        next_id = 0
        
        if not self.continuous_coefficient:
            # Add regular integer coefficients
            for i in range(self.max_coef + 1):
                coef_dict[str(i)] = next_id
                next_id += 1
                
            # Add rational coefficients if provided
            if self.rational_coefficients:
                for num, den in self.rational_coefficients:
                    coef_dict[f"{num}_{den}"] = next_id
                    next_id += 1
        else:
            # In continuous mode, we only need the [C] token ID which is handled separately
            pass
                
        return coef_dict
    
    def _create_special_dict(self) -> Dict[str, int]:
        """Map special tokens to IDs"""
        return {
            '[SEP]': 0,
            '[PAD]': 1,
            '<s>': 2,
            '</s>': 3,
            '+': 4,
            '[BIGSEP]': 5
        }
    
    def _process_monomial(self, monomial: str) -> Union[Tuple[int], ProcessedMonomial]:
        """Convert monomial to (coef_id, pattern_id) and optionally store continuous value"""
        tokens = monomial.strip().split()
        exponents = []
        coef_value = None
        
        for token in tokens:
            if token.startswith('C') or token == '[C]':
                if self.continuous_coefficient:
                    if token == '[C]':
                        coef_value = 1.0  # Default value for [C] token
                    else:
                        coef = token[1:]  # Remove 'C' prefix
                        try:
                            coef_value = float(coef)
                        except ValueError:
                            raise ValueError(f"Invalid continuous coefficient: {coef}")
                    coef_id = self.C_TOKEN_ID
                else:
                    if token == '[C]':
                        raise ValueError("Got [C] token but continuous_coefficient is False")
                    coef = token[1:]  # Remove 'C' prefix
                    coef_id = self.coef_to_id[coef]
            elif token.startswith('E'):
                exponents += [int(token[1:])]  # "E2" -> 2
        
        token_list = [coef_id] + exponents
        
        if self.continuous_coefficient:
            return ProcessedMonomial(tokens=token_list, coefficient_value=coef_value)
        return token_list
    
    def _process_polynomial(self, polynomial: str) -> List[Union[List[int], ProcessedMonomial]]:
        """Convert polynomial to list of processed monomials"""
        if not polynomial.strip():  # If empty string
            return []
        monomials = polynomial.split(' + ')
        special_tokens = [self.special_to_id['+'] for _ in monomials[:-1]] + [self.special_to_id['[SEP]']]
        
        processed = []
        for mono, op in zip(monomials, special_tokens):
            result = self._process_monomial(mono)
            if isinstance(result, ProcessedMonomial):
                result.tokens.append(op)
                processed.append(result)
            else:
                processed.append([*result, op])
        return processed

    def _process(self, text: str) -> List[Union[List[int], ProcessedMonomial]]:
        """Process entire text"""
        bos_tokens = [0] * (self.num_variables + 1) + [self.special_to_id['<s>']]
        bos = ProcessedMonomial(tokens=bos_tokens) if self.continuous_coefficient else bos_tokens
        
        components = text.split(' [BIGSEP] ')
        processed = []
        for component in components:
            polys = component.split(' [SEP] ')
            _processed = [self._process_polynomial(poly) for poly in polys]
            _processed = list(it.chain(*_processed))
            if isinstance(_processed[-1], ProcessedMonomial):
                _processed[-1].tokens[-1] = self.special_to_id['[BIGSEP]']
            else:
                _processed[-1][-1] = self.special_to_id['[BIGSEP]']
            processed.extend(_processed)
        
        if isinstance(processed[-1], ProcessedMonomial):
            processed[-1].tokens[-1] = self.special_to_id['</s>']
        else:
            processed[-1][-1] = self.special_to_id['</s>']
        
        processed = [bos] + processed
        return processed

    def __call__(self, texts: List[str]) -> List[List[Union[List[int], ProcessedMonomial]]]:
        """Process multiple texts"""        
        ret = [self._process(text) for text in texts]
        return ret

    def _decode_monomial_token(self, monomial: Union[torch.Tensor, ProcessedMonomial], skip_special_tokens: bool = False) -> Tuple[str, bool]:
        """Decode monomial token, handling both continuous and discrete cases"""
        if isinstance(monomial, ProcessedMonomial):
            tokens = monomial.tokens
            coeff, exponents, special_id = tokens[0], tokens[1:-1], tokens[-1]
            if isinstance(coeff, torch.Tensor):
                coeff = coeff.item()
            if isinstance(special_id, torch.Tensor):
                special_id = special_id.item()
        else:
            coeff, exponents, special_id = monomial[0].item(), monomial[1:-1], monomial[-1].item()
        
        special_token = self.id_to_special[special_id]
        is_eos = special_token == '</s>'
        
        if special_token == '<s>':
            monomial_text = '' if skip_special_tokens else '<s>'
        else:
            if is_eos and skip_special_tokens:
                special_token = ''
            
            if self.continuous_coefficient:
                if isinstance(monomial, ProcessedMonomial) and monomial.coefficient_value is not None:
                    coef_str = f"{monomial.coefficient_value:.1f}"  # Format as float with 1 decimal place
                else:
                    # During generation, we use C1.0 since we don't predict coefficients
                    coef_str = "1.0"
            else:
                coef_str = self.id_to_coef[coeff]
                
            monomial_text = ' '.join([f'C{coef_str}'] + [f'E{e}' for e in exponents] + [special_token])
         
        return monomial_text.strip(), is_eos
    
    def decode(self, monomial_tokens: Union[torch.Tensor, List[ProcessedMonomial]], skip_special_tokens: bool = False, raise_warning: bool = True) -> str:
        decoded_tokens = []
        is_eos = False
        
        for monomial in monomial_tokens:
            decoded_token, is_eos = self._decode_monomial_token(monomial, skip_special_tokens=skip_special_tokens)
            decoded_tokens.append(decoded_token)
            
            if is_eos:
                break
        
        if (not is_eos) and raise_warning:
            warnings.warn(f'Generation ended before EOS token was found. If you are decoding a generated sequence, the max_length might be too small.')
        
        decoded_text = ' '.join(decoded_tokens).strip()
        return decoded_text
    
    def batch_decode(self, batch_monomial_tokens: List[Union[torch.Tensor, List[ProcessedMonomial]]], skip_special_tokens: bool = True, raise_warning: bool = True) -> List[str]:
        return [self.decode(monomial_tokens, skip_special_tokens=skip_special_tokens, raise_warning=raise_warning) for monomial_tokens in batch_monomial_tokens]

    def is_valid_monomial(self, texts: List[str]) -> List[bool]:
        
        return [self._is_valid_monomial(monomial_text) for monomial_text in texts]
    
    def _is_valid_monomial(self, monomial: str) -> bool:
        items = monomial.split()
        # Accept both C and [C] formats for coefficients
        valid = (items[0].startswith('C') or items[0] == '[C]') and \
                all([t.startswith('E') for t in items[1:-1]]) and \
                items[-1] in self.special_to_id
        return valid

    def generation_helper(self, monomial_texts: List[str]) -> List[str]:
        monomials = [self._generative_helper(monomial_text) for monomial_text in monomial_texts]
        return monomials
    
    def _generative_helper(self, monomial_text: str) -> List[int]:
        """Convert monomial text to list of token IDs for generation"""
        eos = [0] * (self.num_variables + 1) + [self.special_to_id['</s>']]
        
        # Special case: if the text is just a + operator
        if monomial_text.strip() == '+':
            return [0] * (self.num_variables + 1) + [self.special_to_id['+']]
        
        valid = self._is_valid_monomial(monomial_text)

        if valid:
            special_token = monomial_text.split()[-1]
            processed = self._process_monomial(monomial_text)
            if isinstance(processed, ProcessedMonomial):
                monomial = processed.tokens + [self.special_to_id[special_token]]
            else:
                monomial = list(processed) + [self.special_to_id[special_token]]
        else:
            monomial = eos

        return monomial