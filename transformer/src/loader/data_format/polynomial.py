import torch
import itertools as it
import re
from typing import List, Dict, Union
from .standard import StandardDataCollator
from ..data_format.processors.subprocessors import ProcessedMonomial

class MonomialCollator(StandardDataCollator):
    def _pad_sequences(self, sequences: List[Union[List[List[int]], List[ProcessedMonomial]]], padding_value=0):
        """Padding sequences with support for continuous coefficients"""
        max_length = max(len(seq) for seq in sequences)
        
        # Check if we're dealing with continuous coefficients
        is_continuous = isinstance(sequences[0][0], ProcessedMonomial)
        
        if is_continuous:
            num_tokens_per_unit = len(sequences[0][0].tokens)
            padding = ProcessedMonomial(tokens=[0] * num_tokens_per_unit)
        else:
            num_tokens_per_unit = len(sequences[0][0])
            padding = [0] * num_tokens_per_unit

        batch_size = len(sequences)
        sequence_length = max_length
        
        # Prepare result tensors
        monomial_ids = torch.zeros(batch_size, sequence_length, num_tokens_per_unit, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, sequence_length, dtype=torch.long)
        
        # For continuous mode, also prepare coefficient values tensor
        if is_continuous:
            coefficient_values = torch.zeros(batch_size, sequence_length, dtype=torch.float)

        for k, seq in enumerate(sequences):
            cur_length = len(seq)
            padding_length = max_length - cur_length
            padded = seq + [padding] * padding_length
            mask = [1] * cur_length + [0] * padding_length
            
            if is_continuous:
                # Handle ProcessedMonomial instances
                monomial_ids[k, :, :] = torch.tensor([item.tokens for item in padded])
                coefficient_values[k, :] = torch.tensor([
                    item.coefficient_value if item.coefficient_value is not None else 0.0 
                    for item in padded
                ])
            else:
                # Original discrete mode
                monomial_ids[k, :, 0] = torch.tensor([item[0] for item in padded])  # coeff
                monomial_ids[k, :, 1:-1] = torch.tensor([item[1:-1] for item in padded])  # exponents
                monomial_ids[k, :, -1] = torch.tensor([item[-1] for item in padded])  # special
            
            attention_mask[k, :] = torch.tensor(mask)

        result = {
            'monomial_ids': monomial_ids,
            'attention_mask': attention_mask
        }
        
        if is_continuous:
            result['coefficient_values'] = coefficient_values
            
        return result
    
    def __call__(self, batch):
        """Process batch with support for continuous coefficients
        Args:
            batch: Batch obtained from dataset
        
        Returns:
            batch_dict: Dictionary to pass to model
        """
        batch_dict = {}
        attributes = batch[0].keys()
        
        assert(self.tokenizer is not None)        
        assert('input_monomial_ids' in attributes)
        assert('target_monomial_ids' in attributes)
        
        for attribute in attributes:
            attribute_batch = [item[attribute] for item in batch]
            
            if attribute == 'input':
                pass
                
            elif attribute == 'target':
                # For continuous coefficients, we need to replace C1.0 with [C] before tokenization
                processed_batch = [re.sub(r'C[0-9]+\.[0-9]+', '[C]', text) for text in attribute_batch]
                targets = self.tokenizer(processed_batch, padding='longest', return_tensors='pt')
                labels = targets['input_ids'][:, 1:].contiguous()

                if not self.aware_of_padding:
                    labels[labels == self.tokenizer.pad_token_id] = -100
                    
                batch_dict['labels'] = labels
                
            elif 'monomial_ids' in attribute:
                # Process monomial_ids with continuous coefficient support
                prefix = 'decoder_' if attribute.startswith('target_') else ''
                padded = self._pad_sequences(attribute_batch)
                
                batch_dict[f'{prefix}input_ids'] = padded['monomial_ids']
                batch_dict[f'{prefix}attention_mask'] = padded['attention_mask']
                
                # Add coefficient values if present
                if 'coefficient_values' in padded:
                    batch_dict[f'{prefix}coefficient_values'] = padded['coefficient_values']
                
            else:
                if attribute.startswith('target_'):
                    attribute = 'decoder_' + attribute[7:]
                batch_dict[attribute] = self._pad_sequences(attribute_batch, padding_value=0)[:, :-1]
        
        return batch_dict