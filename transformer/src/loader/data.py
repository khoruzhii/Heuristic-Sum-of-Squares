from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Optional, List, Union, Tuple, Iterator
from itertools import islice

import logging
import random
import glob
import orjson
import torch

from torch.utils.data import DataLoader, Dataset

from .data_format.standard import StandardDataset, StandardDataCollator
from .data_format.polynomial import MonomialCollator

@dataclass
class SplitConfig:
    name: str
    batch_size: int
    shuffle: bool = False
    encoding: str = "infix"

def read_split_file(path: str, sample_size: Optional[int] = None, skimming: bool = False) -> tuple[list[str], list[str]]:
    """Read file and split into input and output texts"""
    start_time = time()
    
    try:
        with open(path, "r") as f:
            data = f.read().splitlines()
        if sample_size:
            if skimming:
                random.shuffle(data)
            
            data = data[:sample_size]
            
        input_texts = [line.split(" # ")[0].strip() for line in data]
        target_texts = [line.split(" # ")[1].strip() for line in data]
        
        elapsed = time() - start_time
        logging.info(f"Loaded {len(data)} examples from {path} in {elapsed:.2f} seconds")
        
        return input_texts, target_texts
    
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        raise

def read_split_files(
    dir_path: str,
    prefix: str,
    encoding: str = "infix",
    sample_size: Optional[int] = None,
    skimming: bool = False
) -> tuple[list[str], list[str]]:
    """
    Read multiple files in directory and split into input and output texts
    prefix: e.g. 'train' or 'test'
    encoding: e.g. 'infix'
    """
    pattern = f"{dir_path}_index_range=*.{encoding}"
    files = sorted(glob.glob(pattern))

    all_input_texts = []
    all_target_texts = []
    for file in files:
        print("file:", file)
        input_texts, target_texts = read_split_file(file, None, skimming)
        all_input_texts.extend(input_texts)
        all_target_texts.extend(target_texts)
        if sample_size and len(all_input_texts) >= sample_size:
            break

    if sample_size:
        all_input_texts = all_input_texts[:sample_size]
        all_target_texts = all_target_texts[:sample_size]
        
    return all_input_texts, all_target_texts

def read_polynomial_basis_file_streaming(path: str) -> Iterator[Tuple[str, str]]:
    """Stream read polynomial-basis format file line by line.
    
    Args:
        path (str): Path to the JSON file
        
    Yields:
        Tuple[str, str]: Tuple of polynomial tokens and basis tokens
    """
    with open(path, "rb") as f:  # Note: orjson needs bytes
        for line in f:
            try:
                item = orjson.loads(line)
                polynomial_tokens = " ".join(item["polynomial_tokens"])
                basis_tokens = " ".join(item["basis_tokens"])
                yield polynomial_tokens, basis_tokens
            except Exception as e:
                logging.error(f"Error parsing line: {e}")
                continue

def read_polynomial_basis_file(path: str, sample_size: Optional[int] = None, skimming: bool = False) -> tuple[list[str], list[str]]:
    """Read polynomial-basis format file and split into polynomial and basis tokens.
    Memory efficient version that streams the file.
    
    Args:
        path (str): Path to the JSON file
        sample_size (Optional[int]): Number of examples to load
        skimming (bool): Whether to randomly sample when using sample_size
        
    Returns:
        tuple[list[str], list[str]]: Lists of polynomial tokens and basis tokens
    """
    start_time = time()
    
    try:
        if skimming and sample_size:
            # For skimming with sample size, we need two passes:
            # First to count lines, then to randomly sample
            with open(path, "r") as f:
                total_lines = sum(1 for _ in f)
            
            if total_lines <= sample_size:
                # If sample size is larger than file, just read everything
                data_stream = read_polynomial_basis_file_streaming(path)
            else:
                # Randomly select indices
                selected_indices = set(random.sample(range(total_lines), sample_size))
                # Stream and only keep selected indices
                data_stream = (item for idx, item in enumerate(read_polynomial_basis_file_streaming(path)) 
                             if idx in selected_indices)
        else:
            data_stream = read_polynomial_basis_file_streaming(path)
            if sample_size:
                # Use islice for efficient taking of first n items
                data_stream = islice(data_stream, sample_size)
        
        # Unzip the stream into two lists
        polynomial_tokens, basis_tokens = zip(*data_stream)
        polynomial_tokens, basis_tokens = list(polynomial_tokens), list(basis_tokens)
        
        elapsed = time() - start_time
        logging.info(f"Loaded {len(polynomial_tokens)} polynomial-basis examples from {path} in {elapsed:.2f} seconds")
        
        return polynomial_tokens, basis_tokens
    
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        raise
    except Exception as e:
        logging.error(f"Error reading file {path}: {str(e)}")
        raise

def read_polynomial_basis_files(
    dir_path: str,
    prefix: str,
    sample_size: Optional[int] = None,
    skimming: bool = False
) -> Tuple[List[str], List[str]]:
    """Read multiple polynomial-basis files in directory.
    
    Args:
        dir_path (str): Directory path containing the files
        prefix (str): File prefix to match
        sample_size (Optional[int]): Number of examples to load
        skimming (bool): Whether to randomly sample when using sample_size
        
    Returns:
        Tuple[List[str], List[str]]: Lists of polynomial tokens and basis tokens
    """
    pattern = f"{dir_path}_index_range=*.jsonl"
    files = sorted(glob.glob(pattern))
    
    all_polynomial_tokens = []
    all_basis_tokens = []
    
    for file in files:
        print("file:", file)
        polynomial_tokens, basis_tokens = read_polynomial_basis_file(file, None, skimming)
        all_polynomial_tokens.extend(polynomial_tokens)
        all_basis_tokens.extend(basis_tokens)
        if sample_size and len(all_polynomial_tokens) >= sample_size:
            break
    
    if sample_size:
        all_polynomial_tokens = all_polynomial_tokens[:sample_size]
        all_basis_tokens = all_basis_tokens[:sample_size]
        
    return all_polynomial_tokens, all_basis_tokens

def load_data(
    data_path: Union[str, Path],
    tokenizer = None,
    processor = None,
    subprocessors = None,
    splits: List[SplitConfig] = None,
    return_dataloader: bool = True,
    num_workers: int = None,
    pin_memory: bool = True,
    sample_size: Optional[int] = None,
    train_test_split: Optional[List[int]] = None,
    data_collator_name: Optional[str] = None,
    sample_skimming: bool = False,
    aware_of_padding: bool = False,
    testset_save_path: Optional[str] = None,
    data_format: str = "standard"  # Added parameter for data format
):
    """Function to load data"""
    splits = [SplitConfig(**split) for split in splits]
    assert len(splits) == 1
    split = splits[0]
    
    if data_format == "polynomial_basis":
        pattern = f"{data_path}_index_range=*.jsonl"
        files = sorted(glob.glob(pattern))
        
        sample_size = sum(train_test_split) if train_test_split else sample_size
        
        if files:
            # Use lazy dataset for multiple files
            dataset = LazyPolynomialDataset(
                file_paths=files,
                processor=processor,
                subprocessors=subprocessors,
                sample_size=sample_size
            )
        else:
            # Use lazy dataset for single file
            path = f"{data_path}.jsonl"
            dataset = LazyPolynomialDataset(
                file_paths=[path],
                processor=processor,
                subprocessors=subprocessors,
                sample_size=sample_size
            )
        
        # Handle train/test split for lazy dataset
        if train_test_split is not None:
            train_size, test_size = train_test_split[0], train_test_split[1]
            # Create separate datasets for train and test
            train_dataset = LazyPolynomialDataset(
                file_paths=dataset.file_paths,
                processor=processor,
                subprocessors=subprocessors,
                sample_size=train_size
            )
            test_dataset = LazyPolynomialDataset(
                file_paths=dataset.file_paths,
                processor=processor,
                subprocessors=subprocessors,
                sample_size=test_size
            )
            datasets = [train_dataset, test_dataset]
        else:
            datasets = [dataset]
            
    else:
        # Original standard format handling
        encoding = split.encoding
        pattern = f"{data_path}_index_range=*.{split.encoding}"
        files = glob.glob(pattern)
        
        sample_size = sum(train_test_split) if train_test_split else sample_size
        if files:
            input_texts, target_texts = read_split_files(str(data_path), split.name, encoding, sample_size, skimming=sample_skimming)
        else:
            path = f"{data_path}.{split.encoding}"
            input_texts, target_texts = read_split_file(path, sample_size, skimming=sample_skimming)

        datasets = []
        if train_test_split is not None:
            assert(sum(train_test_split) <= len(input_texts))
            train_size, test_size = train_test_split[0], train_test_split[1]
            input_texts, test_input_texts = input_texts[:train_size], input_texts[train_size:train_size+test_size]
            target_texts, test_target_texts = target_texts[:train_size], target_texts[train_size:train_size+test_size]
            # Create dataset
            dataset = StandardDataset(
                input_texts=input_texts,
                target_texts=target_texts,
                processor=processor,
                subprocessors=subprocessors
            )
            test_dataset = StandardDataset(
                input_texts=test_input_texts,
                target_texts=test_target_texts,
                processor=processor,
                subprocessors=subprocessors
            )
            
            datasets = [dataset, test_dataset]

            if testset_save_path is not None:
                with open(testset_save_path, "w") as f:
                    for inp, tgt in zip(test_input_texts, test_target_texts):
                        f.write(f"{inp} # {tgt}\n")
        else:
            # Create dataset
            dataset = StandardDataset(
                input_texts=input_texts,
                target_texts=target_texts,
                processor=processor,
                subprocessors=subprocessors
            )

            if testset_save_path is not None and split.name == "test":
                with open(testset_save_path, "w") as f:
                    for inp, tgt in zip(input_texts, target_texts):
                        f.write(f"{inp} # {tgt}\n")

            datasets = [dataset]

        
    if data_collator_name in ('standard', None):
        data_collator = StandardDataCollator(tokenizer=tokenizer, aware_of_padding=aware_of_padding)
    elif data_collator_name == 'monomial':
        data_collator = MonomialCollator(tokenizer=tokenizer, aware_of_padding=aware_of_padding)
    else:
        raise ValueError(f"Invalid data collator name: {data_collator_name}")
    
    if return_dataloader:    
        for i, dataset in enumerate(datasets):
            dataloader = DataLoader(
                dataset,
                batch_size=split.batch_size,
                shuffle=split.shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=data_collator
            )
            datasets[i] = dataloader
    
    return datasets if len(datasets) > 1 else datasets[0], data_collator

class LazyPolynomialDataset(Dataset):
    def __init__(self, file_paths, processor, subprocessors, sample_size=None):
        self.file_paths = file_paths
        self.processor = processor
        self.subprocessors = subprocessors if subprocessors is not None else {}
        self.sample_size = sample_size
        self._length = None
        self._line_offsets = None  # Cache line offsets for fast access
        
    def __len__(self):
        if self._length is None:
            # Count lines without loading data
            self._length = sum(self._count_lines(f) for f in self.file_paths)
            if self.sample_size:
                self._length = min(self._length, self.sample_size)
        return self._length
    
    def __getitem__(self, idx):
        # Use pre-computed offsets for fast access
        return self._load_item_fast(idx)
    
    def _count_lines(self, path):
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    
    def _compute_line_offsets(self):
        """Pre-compute byte offsets for all lines in all files"""
        if self._line_offsets is not None:
            return self._line_offsets
            
        self._line_offsets = []
        for file_path in self.file_paths:
            file_offsets = [0]  # Start of file
            with open(file_path, "rb") as f:
                for line in f:
                    file_offsets.append(f.tell())
            self._line_offsets.append(file_offsets)
        return self._line_offsets
    
    def _load_item_fast(self, idx):
        """Load item using pre-computed offsets for O(1) access"""
        if self._line_offsets is None:
            self._compute_line_offsets()
        
        # Find which file contains this index
        current_idx = 0
        for file_idx, file_path in enumerate(self.file_paths):
            file_length = len(self._line_offsets[file_idx]) - 1  # -1 because offsets include end-of-file
            
            if current_idx + file_length > idx:
                # This file contains our index
                local_idx = idx - current_idx
                if local_idx >= file_length:
                    raise IndexError(f"Index {idx} out of range")
                
                # Direct seek to the line
                with open(file_path, "rb") as f:
                    f.seek(self._line_offsets[file_idx][local_idx])
                    line = f.readline().decode('utf-8').strip()
                    item = orjson.loads(line)
                    return self._process_single_item(item)
            
            current_idx += file_length
        
        raise IndexError(f"Index {idx} out of range")
    
    def _process_single_item(self, item):
        """Process a single JSON item into the expected format"""
        # Extract polynomial and basis tokens from JSON
        polynomial_tokens = " ".join(item["polynomial_tokens"])
        basis_tokens = " ".join(item["basis_tokens"])
        
        # Apply processor if it exists (usually empty for polynomial_basis format)
        if self.processor is not None:
            polynomial_tokens = self.processor([polynomial_tokens], is_target=False)[0]
            basis_tokens = self.processor([basis_tokens], is_target=True)[0]
        
        # Create base item
        processed_item = {
            "input": polynomial_tokens,
            "target": basis_tokens,
        }
        
        # Apply subprocessors (like MonomialProcessorPlus)
        for processor_name, subprocessor in self.subprocessors.items():
            input_attribute = subprocessor([polynomial_tokens])[0]
            target_attribute = subprocessor([basis_tokens])[0]
            
            processed_item[f'input_{processor_name}'] = input_attribute
            processed_item[f'target_{processor_name}'] = target_attribute
        
        return processed_item