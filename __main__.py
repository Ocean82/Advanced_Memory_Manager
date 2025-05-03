"""
Advanced Memory Management System
---------------------------------
A high-performance, lossless data storage and retrieval system that optimizes memory usage
through intelligent compression, deduplication, and tiered storage.

Features:
- Multiple compression algorithms with automatic selection
- Content-aware deduplication
- Tiered storage (memory, memory-mapped files, disk)
- Background optimization and eviction
- Thread-safe operations
- Detailed statistics and monitoring
- Asynchronous operations
- Caching mechanisms
- Advanced metadata management
"""

import os
import io
import time
import hashlib
import sqlite3
import threading
import json
import mmap
import shutil
import pickle
import msgpack
import numpy as np
import blosc2
import zstandard as zstd
import lz4.frame
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any, Set, BinaryIO
from dataclasses import dataclass, field, asdict
import psutil
import asyncio
from functools import lru_cache
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as feather
import sys
import types
import zlib
from collections import OrderedDict
import logging
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('memory_manager.log')
    ]
)
logger = logging.getLogger('AdvancedMemoryManager')

class CompressionEngine:
    """Handles data compression and decompression using multiple algorithms."""

    ALGORITHMS = ['zstd', 'lz4', 'blosclz', 'zlib', 'blosc2']

    @staticmethod
    def _pad_data_for_blosc(data: bytes, typesize: int = 8) -> bytes:
        """
        Pad data to be a multiple of typesize for blosc compression.
        
        Args:
            data: The data to pad
            typesize: The typesize to align to
            
        Returns:
            Padded data
        """
        remainder = len(data) % typesize
        if remainder == 0:
            return data
        
        # Add padding to make it a multiple of typesize
        padding_size = typesize - remainder
        return data + (b'\0' * padding_size)

    @staticmethod
    def compress(data: bytes, algorithm: str, compression_level: int = 6) -> Tuple[bytes, int]:
        """
        Compress data using the specified algorithm.

        Args:
            data: The data to compress as bytes
            algorithm: Compression algorithm to use
            compression_level: Compression level (1-9, higher = better compression but slower)

        Returns:
            Tuple of (compressed_data, original_size)

        Raises:
            ValueError: If algorithm is not supported
        """
        if algorithm not in CompressionEngine.ALGORITHMS:
            raise ValueError(f"Unknown compression algorithm: {algorithm}. "
                             f"Supported algorithms: {', '.join(CompressionEngine.ALGORITHMS)}")

        original_size = len(data)

        try:
            if algorithm == 'zstd':
                cctx = zstd.ZstdCompressor(level=compression_level)
                compressed = cctx.compress(data)
            elif algorithm == 'lz4':
                compressed = lz4.frame.compress(data, compression_level=compression_level)
            elif algorithm in ['blosclz', 'blosc2']:
                # Pad data to be a multiple of typesize
                typesize = 8  # Using 8 as default typesize
                padded_data = CompressionEngine._pad_data_for_blosc(data, typesize)
                
                # Use blosc2 directly
                compressed = blosc2.compress(
                    padded_data,
                    typesize=typesize,
                    clevel=compression_level
                )
            elif algorithm == 'zlib':
                compressed = zlib.compress(data, level=compression_level)
            else:
                raise ValueError(f"Algorithm implementation missing: {algorithm}")

            return compressed, original_size
        except Exception as e:
            logger.error(f"Compression failed with algorithm '{algorithm}': {str(e)}")
            raise ValueError(f"Compression failed with algorithm '{algorithm}': {str(e)}")

    @staticmethod
    def _compress_sample(data: bytes, algorithm: str, compression_level: int) -> Tuple[bytes, int]:
        """Internal method for sample compression using the compress method."""
        try:
            return CompressionEngine.compress(data, algorithm, compression_level)
        except Exception as e:
            # Log the error but don't raise it - this allows testing to continue with other algorithms
            logger.warning(f"Error testing compression algorithm {algorithm}: {str(e)}")
            # Return uncompressed data and original size as fallback
            return data, len(data)

    @staticmethod
    def decompress(compressed_data: bytes, algorithm: str, original_size: Optional[int] = None) -> bytes:
        """
        Decompress data that was compressed with a supported algorithm.

        Args:
            compressed_data: The compressed data
            algorithm: Compression algorithm used
            original_size: Expected original size (for verification)

        Returns:
            Decompressed data as bytes

        Raises:
            ValueError: If algorithm is not supported or size verification fails
        """
        if algorithm not in CompressionEngine.ALGORITHMS:
            raise ValueError(f"Unknown compression algorithm: {algorithm}")

        try:
            if algorithm == 'zstd':
                dctx = zstd.ZstdDecompressor()
                decompressed = dctx.decompress(compressed_data)
            elif algorithm == 'lz4':
                decompressed = lz4.frame.decompress(compressed_data)
            elif algorithm in ['blosclz', 'blosc2']:
                decompressed = blosc2.decompress(compressed_data)
            elif algorithm == 'zlib':
                decompressed = zlib.decompress(compressed_data)
            else:
                raise ValueError(f"Algorithm implementation missing: {algorithm}")

            # Verify the size if original_size was provided
            if original_size is not None and len(decompressed) != original_size:
                # If we padded the data for blosc, we need to truncate it back
                if algorithm in ['blosclz', 'blosc2'] and len(decompressed) > original_size:
                    decompressed = decompressed[:original_size]
                elif len(decompressed) != original_size:
                    raise ValueError(f"Decompressed size {len(decompressed)} doesn't match original size {original_size}")

            return decompressed
        except Exception as e:
            logger.error(f"Decompression failed with algorithm '{algorithm}': {str(e)}")
            raise ValueError(f"Decompression failed with algorithm '{algorithm}': {str(e)}")


class SecurityError(Exception):
    """Exception raised when attempting unsafe deserialization of untrusted data."""
    pass


class Serializer:
    """
    Handles serialization and deserialization of Python objects.

    Security Warning:
    This serializer can use pickle and numpy.load with allow_pickle=True,
    which can lead to arbitrary code execution if deserializing untrusted data.
    Always use the trusted=True parameter only with data from verified sources.
    """

    # Supported serialization formats
    FORMATS = ['pickle', 'numpy', 'json', 'raw', 'msgpack', 'parquet', 'feather', 'auto']

    @staticmethod
    def is_numpy_compatible(data: Any) -> bool:
        """
        Check if data can be efficiently serialized as a NumPy array.

        Args:
            data: Data to check

        Returns:
            True if data is a NumPy array or can be efficiently converted to one
        """
        # Check if it's already a numpy array
        if isinstance(data, np.ndarray):
            return True

        # Check for types that can be efficiently converted to numpy arrays
        if isinstance(data, (list, tuple)):
            if not data:
                return False

            # Check if all elements are of the same type
            first_type = type(data[0])
            if all(isinstance(item, first_type) for item in data):
                # Check if the type is a simple numeric type
                if first_type in (int, float, bool, np.number):
                    return True

        return False

    @staticmethod
    def validate_format(data: bytes, format: str) -> bool:
        """
        Verify that the data matches the claimed format.

        Args:
            data: Serialized data
            format: Format to validate against

        Returns:
            True if data appears to match the format, False otherwise
        """
        try:
            if format == 'numpy':
                # Check for numpy header signature
                return data[:6] == b'\x93NUMPY'

            elif format == 'json':
                # Check for JSON structure
                decoded = data.decode('utf-8', errors='ignore').strip()
                return (decoded.startswith('{') and decoded.endswith('}')) or \
                    (decoded.startswith('[') and decoded.endswith(']'))

            elif format == 'pickle':
                # Limited validation - check pickle protocol marker
                return data[0:1] in (b'\x80', b'\x81', b'\x82', b'\x83', b'\x84', b'\x85')

            elif format == 'msgpack':
                # Basic check - msgpack has no clear header
                return True

            elif format == 'parquet':
                # Check for PAR1 magic number
                return data[0:4] == b'PAR1' and data[-4:] == b'PAR1'

            elif format == 'feather':
                # Check for feather magic bytes
                return data[0:4] == b'FEA1' or data[0:8] == b'ARROW1\x00\x00'

            elif format == 'raw':
                # Raw format is just bytes, always valid
                return True

            return False
        except Exception:
            return False

    @staticmethod
    def serialize(
            data: Any,
            format: Optional[str] = None,
            compress: bool = False,
            verify_integrity: bool = False
    ) -> Tuple[bytes, str, bool, Optional[str]]:
        """
        Serialize data to bytes using the most appropriate format.

        Args:
            data: Data to serialize
            format: Format to use (if None or 'auto', best format is automatically selected)
            compress: Whether to compress the serialized data
            verify_integrity: Whether to compute and return a hash of the data

        Returns:
            Tuple of (serialized_data, format_used, is_compressed, integrity_hash)

        Raises:
            ValueError: If the format is not supported
            TypeError: If the data cannot be serialized in the specified format
        """
        # Auto-detect format if not specified or if format is 'auto'
        if format is None or format == 'auto':
            # For numpy arrays, use numpy format
            if isinstance(data, np.ndarray):
                format = 'numpy'
            # For raw bytes, use raw format
            elif isinstance(data, bytes):
                format = 'raw'
            # For tabular data, use parquet
            elif isinstance(data, (list, tuple)) and len(data) > 0 and all(isinstance(item, dict) for item in data):
                format = 'parquet'
            # For simple types that can be JSON serialized, use JSON
            elif isinstance(data, (dict, list, str, int, float, bool)) and not isinstance(data, bytes):
                try:
                    json.dumps(data)  # Test if JSON serializable
                    format = 'json'
                except (TypeError, OverflowError):
                    format = 'pickle'
            # For complex Python objects, use pickle
            elif isinstance(data, (set, frozenset, complex, slice, range, type)) or \
                    isinstance(data, types.FunctionType) or \
                    isinstance(data, type(sys)) or \
                    isinstance(data, types.ModuleType) or \
                    isinstance(data, types.MethodType):
                format = 'pickle'
            # For data that could use msgpack (more efficient than JSON for some structures)
            elif isinstance(data, (dict, list)):
                if isinstance(data, list):
                    # Check if list contains no nested structures
                    has_no_nested = not any(isinstance(item, (dict, list)) for item in data)
                    if has_no_nested:
                        format = 'msgpack'
                else:  # It's a dict
                    format = 'msgpack'
            # Default to pickle for everything else
            else:
                format = 'pickle'

        # Validate format
        if format not in Serializer.FORMATS:
            raise ValueError(f"Unsupported serialization format: {format}")

        # Serialize based on the selected format
        try:
            if format == 'numpy':
                # Handle special case for list conversion to numpy
                if isinstance(data, (list, tuple)) and Serializer.is_numpy_compatible(data):
                    data = np.array(data)

                with io.BytesIO() as buffer:
                    np.save(buffer, data)
                    serialized = buffer.getvalue()

            elif format == 'json':
                serialized = json.dumps(data).encode('utf-8')

            elif format == 'pickle':
                serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

            elif format == 'raw':
                if isinstance(data, bytes):
                    serialized = data
                elif isinstance(data, (bytearray, memoryview)):
                    serialized = bytes(data)
                else:
                    raise TypeError(f"Raw format requires bytes-like object, got {type(data).__name__}")

            elif format == 'msgpack':
                serialized = msgpack.packb(data, use_bin_type=True)

            elif format == 'parquet' or format == 'feather':
                with io.BytesIO() as buffer:
                    table = pa.Table.from_pylist(data if isinstance(data, list) else [data])

                    if format == 'parquet':
                        pq.write_table(table, buffer)
                    else:  # feather
                        feather.write_feather(table, buffer)

                    serialized = buffer.getvalue()

            # Apply compression if requested
            is_compressed = False
            if compress and serialized:
                compressed, algo, ratio = compress_data(serialized)
                serialized = compressed
                is_compressed = True

            # Calculate hash if integrity verification is requested
            data_hash = None
            if verify_integrity:
                data_hash = hashlib.sha256(serialized).hexdigest()

            return serialized, format, is_compressed, data_hash

        except Exception as e:
            logger.error(f"Failed to serialize data using {format} format: {str(e)}")
            raise TypeError(f"Failed to serialize data using {format} format: {str(e)}")

    @staticmethod
    def deserialize(
            data: bytes,
            format: str,
            trusted: bool = False,
            compressed: bool = False,
            original_hash: Optional[str] = None
    ) -> Any:
        """
        Deserialize bytes back to the original data.

        Args:
            data: Serialized data
            format: Format used for serialization
            trusted: Set to True only if data comes from a trusted source
            compressed: Whether the data is compressed
            original_hash: If provided, verify data integrity against this hash

        Returns:
            Deserialized data

        Raises:
            ValueError: If the format is not supported
            SecurityError: If untrusted data is being loaded with an unsafe format
            TypeError: If the data cannot be deserialized from the specified format
        """
        if format not in Serializer.FORMATS or format == 'auto':
            raise ValueError(f"Unsupported deserialization format: {format}")

        # Decompress if needed
        if compressed:
            try:
                data = zlib.decompress(data)
            except zlib.error as e:
                raise ValueError(f"Failed to decompress data: {str(e)}")

        # Verify integrity if hash is provided
        if original_hash:
            computed_hash = hashlib.sha256(data).hexdigest()
            if computed_hash != original_hash:
                raise SecurityError("Data integrity check failed: hash mismatch")

        # Validate format (skip 'auto' since it's not a real format for deserialization)
        if format != 'auto' and not Serializer.validate_format(data, format):
            raise ValueError(f"Data does not appear to match the claimed format: {format}")

        # Security check for unsafe formats
        if not trusted and format in ['pickle', 'numpy']:
            raise SecurityError(
                f"Cannot deserialize untrusted data with unsafe format: {format}. "
                "Set trusted=True only if the data comes from a verified source."
            )

        try:
            if format == 'numpy':
                with io.BytesIO(data) as buffer:
                    try:
                        # First try without allow_pickle
                        result = np.load(buffer, allow_pickle=False)
                    except ValueError:
                        # If that fails and data is trusted, try with allow_pickle
                        if trusted:
                            buffer.seek(0)
                            result = np.load(buffer, allow_pickle=True)
                        else:
                            raise SecurityError(
                                "Cannot load NumPy data that requires pickle unpickling with trusted=False")
                    return result

            elif format == 'json':
                try:
                    return json.loads(data.decode('utf-8'))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON data: {str(e)}")

            elif format == 'pickle':
                try:
                    return pickle.loads(data)
                except pickle.UnpicklingError as e:
                    raise ValueError(f"Invalid pickle data: {str(e)}")

            elif format == 'raw':
                return data

            elif format == 'msgpack':
                try:
                    return msgpack.unpackb(data, raw=False)
                except Exception as e:
                    raise ValueError(f"Invalid msgpack data: {str(e)}")

            elif format == 'parquet':
                with io.BytesIO(data) as buffer:
                    try:
                        table = pq.read_table(buffer)
                        return table.to_pylist()
                    except Exception as e:
                        raise ValueError(f"Invalid parquet data: {str(e)}")

            elif format == 'feather':
                with io.BytesIO(data) as buffer:
                    try:
                        table = feather.read_feather(buffer)
                        return table.to_pylist()
                    except Exception as e:
                        raise ValueError(f"Invalid feather data: {str(e)}")

        except (SecurityError, ValueError) as e:
            # Re-raise these specific exceptions
            raise
        except Exception as e:
            raise TypeError(f"Failed to deserialize data using {format} format: {str(e)}")


def compress_data(data: Union[bytes, np.ndarray, Any],
             algorithm: str = 'auto',
             compression_level: int = 6) -> Tuple[bytes, str, float]:
    """
    Compress data using the specified algorithm.

    Args:
        data: Data to compress (bytes, numpy array, or object implementing Serializable)
        algorithm: Compression algorithm to use ('auto' or one of CompressionEngine.ALGORITHMS)
        compression_level: Compression level (1-9, higher = better compression but slower)

    Returns:
        Tuple of (compressed_data, algorithm_used, compression_ratio)

    Raises:
        ValueError: If algorithm is not supported or compression fails
        TypeError: If data cannot be properly converted to bytes
    """
    # First, check if algorithm is valid
    if algorithm != 'auto' and algorithm not in CompressionEngine.ALGORITHMS:
        raise ValueError(f"Unsupported compression algorithm: {algorithm}. "
                         f"Supported algorithms: {', '.join(CompressionEngine.ALGORITHMS)} or 'auto'")

    # Convert data to bytes depending on its type
    try:
        if isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        elif isinstance(data, bytes):
            data_bytes = data
        elif hasattr(data, '__bytes__'):
            # Use __bytes__ method if available
            data_bytes = bytes(data)
        else:
            # For other types, use serialization
            data_bytes = Serializer.serialize(data, 'pickle')[0]
    except Exception as e:
        raise TypeError(f"Failed to convert data to bytes: {str(e)}")

    # Get original size for compression ratio calculation
    original_size = len(data_bytes)

    # Handle empty data case
    if original_size == 0:
        return b'', 'zstd', 1.0  # No compression needed for empty data

    try:
        if algorithm == "auto":
            # For larger datasets, intelligently sample a portion for algorithm testing
            if original_size > 10 * 1024 * 1024:  # > 10MB
                sample_size = min(original_size // 10, 5 * 1024 * 1024)  # 10% or up to 5MB
            else:
                sample_size = min(original_size, 1024 * 1024)  # All or up to 1MB

            # Use stratified sampling for large datasets to get more representative sample
            if original_size > sample_size * 2:
                # Take chunks from beginning, middle and end
                chunk_size = sample_size // 3
                sample = data_bytes[:chunk_size] + data_bytes[
                                               original_size // 2 - chunk_size // 2:original_size // 2 + chunk_size // 2] + data_bytes[
                                                                                                                            -chunk_size:]
            else:
                sample = data_bytes[:sample_size]

            best_ratio = 0
            best_algo = "zstd"  # Default
            compressed_results = {}  # Store compressed results to avoid recompression

            # Test all compression algorithms on sample
            for algo in CompressionEngine.ALGORITHMS:
                try:
                    compressed_sample, _ = CompressionEngine._compress_sample(sample, algo, compression_level)
                    # Skip if compression failed (the _compress_sample method now returns the original data on failure)
                    if len(compressed_sample) == len(sample):
                        continue
                        
                    sample_ratio = len(sample) / len(compressed_sample) if len(compressed_sample) > 0 else 1

                    if sample_ratio > best_ratio:
                        best_ratio = sample_ratio
                        best_algo = algo

                    # If compression is better than 1.1x, store the result
                    if sample_ratio > 1.1:
                        compressed_results[algo] = compressed_sample
                except Exception as e:
                    # Log the error but continue with other algorithms
                    logger.warning(f"Error testing compression algorithm {algo}: {str(e)}")
                    continue

            algorithm = best_algo

            # Full compression with selected algorithm
            try:
                compressed, _ = CompressionEngine.compress(data_bytes, algorithm, compression_level)
            except Exception as e:
                # If the best algorithm fails, try the next best one
                logger.warning(f"Compression failed with algorithm '{algorithm}': {str(e)}")
                fallback_algos = [a for a in CompressionEngine.ALGORITHMS if a != algorithm]
                for fallback in fallback_algos:
                    try:
                        logger.info(f"Trying fallback algorithm: {fallback}")
                        compressed, _ = CompressionEngine.compress(data_bytes, fallback, compression_level)
                        algorithm = fallback  # Update the algorithm to the one that worked
                        break
                    except Exception as e2:
                        logger.warning(f"Fallback compression failed with algorithm '{fallback}': {str(e2)}")
                else:
                    # If all algorithms fail, return uncompressed data
                    logger.warning("All compression algorithms failed, returning uncompressed data")
                    return data_bytes, 'raw', 1.0
            
        else:
            # Direct compression with specified algorithm
            try:
                compressed, _ = CompressionEngine.compress(data_bytes, algorithm, compression_level)
            except Exception as e:
                # If specified algorithm fails, return uncompressed data
                logger.warning(f"Compression failed with algorithm '{algorithm}': {str(e)}")
                return data_bytes, 'raw', 1.0

        # Calculate compression ratio, handling potential division by zero
        if len(compressed) > 0:
            compression_ratio = original_size / len(compressed)
        else:
            compression_ratio = 1.0  # Default in case of empty compressed data

        return compressed, algorithm, compression_ratio

    except Exception as e:
        # Last resort fallback - return uncompressed data
        logger.error(f"Compression completely failed: {str(e)}")
        return data_bytes, 'raw', 1.0


@dataclass
class MemoryBlock:
    """Represents a block of data stored in the system."""
    key: str
    data_hash: str
    compression_algo: str
    serialization_format: str
    original_size: int
    compressed_size: int
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    storage_location: str = "memory"  # "memory", "disk", or "hybrid"
    path: Optional[str] = None
    mmap_start: Optional[int] = None
    mmap_size: Optional[int] = None


class StorageManager:
    """Manages the different storage tiers: memory, memory-mapped files, and disk."""

    def __init__(self,
                 disk_path: str = "j:/MyStorage5/cache",
                 memory_limit: int = 1024 * 1024 * 256,  # 256 MB default
                 mmap_size: int = 1024 * 1024 * 512):     # 512 MB default
        """
        Initialize the storage manager.

        Args:
            disk_path: Path to store data on disk
            memory_limit: Maximum memory usage in bytes
            mmap_size: Size of the memory-mapped file in bytes
        """
        self.memory_store: Dict[str, bytes] = {}
        self.memory_limit = memory_limit
        self.current_memory_usage = 10

        # Setup disk storage
        self.disk_path = Path(disk_path)
        self.disk_path.mkdir(parents=True, exist_ok=True)

        # Setup memory-mapped file
        self.mmap_file_path = self.disk_path / "mmap_storage.bin"
        self.mmap_size = mmap_size
        self.lock = threading.RLock()
        
        # Initialize mmap file and object
        self.mmap_file = None
        self.mmap = None

        # Initialize mmap
        self._init_mmap()

    def _init_mmap(self):
        """Initialize the memory-mapped file storage."""
        with self.lock:
            try:
                # Create or open the mmap file
                if not self.mmap_file_path.exists():
                    # Create a new empty file of the specified size
                    with open(self.mmap_file_path, 'wb') as f:
                        f.write(b'\0' * self.mmap_size)

                # Open the mmap file
                self.mmap_file = open(self.mmap_file_path, 'r+b')

                # Create the mmap object
                self.mmap = mmap.mmap(
                    self.mmap_file.fileno(),
                    self.mmap_size,
                    access=mmap.ACCESS_WRITE
                )

                # Initialize index for mmap allocations
                # Format: {key: (start_pos, size)}
                self.mmap_index = {}
                
                logger.info(f"Initialized memory-mapped file at {self.mmap_file_path} with size {self.mmap_size} bytes")
            except Exception as e:
                logger.error(f"Failed to initialize memory-mapped file: {str(e)}")
                # Ensure resources are cleaned up
                if hasattr(self, 'mmap') and self.mmap:
                    try:
                        self.mmap.close()
                    except:
                        pass
                    self.mmap = None
                
                if hasattr(self, 'mmap_file') and self.mmap_file:
                    try:
                        self.mmap_file.close()
                    except:
                        pass
                    self.mmap_file = None
                
                # Re-raise the exception
                raise

    def _resize_mmap(self, new_size: int):
        """
        Resize the memory-mapped file to accommodate more data.

        Args:
            new_size: New size for the mmap file in bytes
        """
        with self.lock:
            try:
                logger.info(f"Resizing memory-mapped file from {self.mmap_size} to {new_size} bytes")
                
                # Close existing mmap
                if self.mmap:
                    self.mmap.close()
                    self.mmap = None
                
                if self.mmap_file:
                    self.mmap_file.close()
                    self.mmap_file = None

                # Resize the file
                with open(self.mmap_file_path, 'wb') as f:
                    f.write(b'\0' * new_size)

                # Reopen mmap with new size
                self.mmap_file = open(self.mmap_file_path, 'r+b')
                self.mmap = mmap.mmap(
                    self.mmap_file.fileno(),
                    new_size,
                    access=mmap.ACCESS_WRITE
                )

                self.mmap_size = new_size
                logger.info(f"Memory-mapped file resized successfully to {new_size} bytes")
            except Exception as e:
                logger.error(f"Failed to resize memory-mapped file: {str(e)}")
                # Ensure resources are cleaned up
                
            if hasattr(self, 'mmap_file') and self.mmap_file:
                try:
                    self.mmap_file.close()
                except Exception as e:
                # Handle or log the exception if needed
                    pass
                finally:
                    self.mmap_file = None
                
                # Try to reinitialize with original size
                try:
                    self._init_mmap()
                except:
                    pass
                
                # Re-raise the exception
                raise

    def store_in_memory(self, key: str, data: bytes) -> int:
        """
        Store data in memory.

        Args:
            key: Key to associate with the data
            data: Data to store

        Returns:
            Size of stored data in bytes
        """
        with self.lock:
            # Store data in memory
            self.memory_store[key] = data
            size = len(data)
            self.current_memory_usage += size
            return size

    def store_on_disk(self, key: str, data: bytes) -> Tuple[str, int]:
        """
        Store data on disk.

        Args:
            key: Key to associate with the data
            data: Data to store

        Returns:
            Tuple of (file_path, size)
        """
        with self.lock:
            try:
                # Create a safe file path from the key
                safe_key = hashlib.md5(key.encode()).hexdigest()
                file_path = self.disk_path / f"{safe_key}.bin"

                # Write data to file
                with open(file_path, 'wb') as f:
                    f.write(data)

                return str(file_path), len(data)
            except Exception as e:
                logger.error(f"Failed to store data on disk for key {key}: {str(e)}")
                raise

    def store_in_hybrid(self, key: str, data: bytes) -> Tuple[str, int, int]:
        """
        Store data in hybrid mode (mmap + disk overflow).

        Args:
            key: Key to associate with the data
            data: Data to store

        Returns:
            Tuple of (file_path, start_pos, size)
        """
        size = len(data)

        with self.lock:
            try:
                # Check if we need to resize the mmap file
                if size > self.mmap_size:
                    # If data is too large, store directly on disk
                    file_path, _ = self.store_on_disk(key, data)
                    return file_path, -1, size

                # Find a suitable position in the mmap file
                # Check if mmap has enough free space
                total_used = sum(size for _, size in self.mmap_index.values())
                free_space = self.mmap_size - total_used

                if free_space < size:
                    # Not enough space, resize the mmap file
                    new_size = max(self.mmap_size * 2, self.mmap_size + size)
                    self._resize_mmap(new_size)

                # Simple allocation strategy: place at the end
                start_pos = total_used

                # Store data in mmap
                self.mmap[start_pos:start_pos + size] = data

                # Update the index
                self.mmap_index[key] = (start_pos, size)

                return str(self.mmap_file_path), start_pos, size
            except Exception as e:
                logger.error(f"Failed to store data in hybrid storage for key {key}: {str(e)}")
                # Fallback to disk storage
                try:
                    file_path, _ = self.store_on_disk(key, data)
                    return file_path, -1, size
                except Exception as e2:
                    logger.error(f"Fallback to disk storage also failed: {str(e2)}")
                    raise ValueError(f"Failed to store data in any storage location: {str(e)}, then {str(e2)}")

    def retrieve_from_memory(self, key: str) -> bytes:
        """
        Retrieve data from memory.

        Args:
            key: Key associated with the data

        Returns:
            The retrieved data
        """
        with self.lock:
            if key not in self.memory_store:
                raise KeyError(f"Key not found in memory storage: {key}")

            return self.memory_store[key]

    def retrieve_from_disk(self, file_path: str) -> bytes:
        """
        Retrieve data from disk.

        Args:
            file_path: Path where the data is stored

        Returns:
            The retrieved data
        """
        with self.lock:
            try:
                with open(file_path, 'rb') as f:
                    return f.read()
            except FileNotFoundError:
                raise KeyError(f"File not found on disk: {file_path}")
            except Exception as e:
                logger.error(f"Failed to retrieve data from disk at {file_path}: {str(e)}")
                raise

    def retrieve_from_hybrid(self,
                             file_path: str,
                             start_pos: int,
                             size: int) -> bytes:
        """
        Retrieve data from hybrid storage.

        Args:
            file_path: Path to the storage file
            start_pos: Starting position in the file (-1 for disk-only storage)
            size: Size of the data to retrieve

        Returns:
            The retrieved data
        """
        with self.lock:
            try:
                # If start_pos is -1, it means the data is stored on disk
                if start_pos == -1:
                    return self.retrieve_from_disk(file_path)

                # Otherwise, retrieve from mmap
                if not self.mmap:
                    raise ValueError("Memory-mapped file is not initialized")
                
                return bytes(self.mmap[start_pos:start_pos + size])
            except Exception as e:
                logger.error(f"Failed to retrieve data from hybrid storage: {str(e)}")
                raise

    def remove_data(self,
                    key: str,
                    storage_location: str,
                    path: Optional[str] = None,
                    mmap_position: Optional[Tuple[int, int]] = None) -> bool:
        """
        Remove data from the specified storage.

        Args:
            key: Key associated with the data
            storage_location: Location where the data is stored ("memory", "disk", or "hybrid")
            path: Path to the data file (for disk or hybrid storage)
            mmap_position: Position in mmap file (for hybrid storage)

        Returns:
            True if data was successfully removed
        """
        with self.lock:
            try:
                if storage_location == "memory":
                    if key in self.memory_store:
                        data_size = len(self.memory_store[key])
                        del self.memory_store[key]
                        self.current_memory_usage -= data_size
                        return True
                    return False

                elif storage_location == "disk":
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                            return True
                        except OSError as e:
                            logger.error(f"Failed to remove file at {path}: {str(e)}")
                            return False
                    return False

                elif storage_location == "hybrid":
                    # Remove from mmap index if present
                    if key in self.mmap_index:
                        del self.mmap_index[key]

                    # If data was stored on disk due to size, remove the file
                    if path and path != str(self.mmap_file_path) and os.path.exists(path):
                        try:
                            os.remove(path)
                            return True
                        except OSError as e:
                            logger.error(f"Failed to remove hybrid file at {path}: {str(e)}")
                            return False

                    return True

                return False
            except Exception as e:
                logger.error(f"Error removing data for key {key}: {str(e)}")
                return False

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the storage usage.

        Returns:
            Dictionary with storage statistics
        """
        with self.lock:
            try:
                # Memory stats
                memory_usage = self.current_memory_usage
                memory_limit = self.memory_limit
                memory_usage_pct = (memory_usage / memory_limit) * 100 if memory_limit > 0 else 10

                # Disk stats
                disk_usage = sum(os.path.getsize(f) for f in self.disk_path.glob("*.bin") 
                                if f != self.mmap_file_path and os.path.exists(f))

                # Mmap stats
                mmap_usage = sum(size for _, size in self.mmap_index.values())
                mmap_limit = self.mmap_size
                mmap_usage_pct = (mmap_usage / mmap_limit) * 100 if mmap_limit > 0 else 0

                # Total stats
                total_usage = memory_usage + disk_usage + mmap_usage

                return {
                    "memory": {
                        "usage": memory_usage,
                        "limit": memory_limit,
                        "usage_pct": memory_usage_pct,
                        "items": len(self.memory_store)
                    },
                    "disk": {
                        "usage": disk_usage,
                        "items": len(list(self.disk_path.glob("*.bin"))) - 1  # Exclude mmap file
                    },
                    "mmap": {
                        "usage": mmap_usage,
                        "limit": mmap_limit,
                        "usage_pct": mmap_usage_pct,
                        "items": len(self.mmap_index)
                    },
                    "total": {
                        "usage": total_usage
                    }
                }
            except Exception as e:
                logger.error(f"Error getting storage stats: {str(e)}")
                # Return minimal stats in case of error
                return {
                    "memory": {"usage": self.current_memory_usage, "limit": self.memory_limit},
                    "error": str(e)
                }

    def cleanup(self):
        """Clean up all resources properly."""
        with self.lock:
            # Close mmap
            try:
                if hasattr(self, 'mmap') and self.mmap:
                    self.mmap.close()
                    self.mmap = None
            except Exception as e:
                logger.error(f"Error closing mmap: {e}")

            # Close mmap file
            try:
                if hasattr(self, 'mmap_file') and self.mmap_file:
                    self.mmap_file.close()
                    self.mmap_file = None
            except Exception as e:
                logger.error(f"Error closing mmap file: {e}")

            # Clear memory store
            try:
                self.memory_store.clear()
                self.current_memory_usage = 0
            except Exception as e:
                logger.error(f"Error clearing memory store: {e}")


class MetadataManager:
    """Manages metadata for stored objects using SQLite database."""

    def __init__(self, db_path: str = ".cache/metadata.db"):
        """
        Initialize the metadata manager.

        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = Path(db_path)

        # Make sure the directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.conn = None
        self.connect_to_db()

        # Thread safety
        self.lock = threading.RLock()

        # Initialize database tables
        self._init_db()

    def connect_to_db(self):
        """Connect to the SQLite database with proper error handling."""
        try:
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL")  # Use Write-Ahead Logging for better concurrency
            self.conn.execute("PRAGMA synchronous=NORMAL")  # Slightly faster with good safety
            logger.info(f"Connected to SQLite database at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to SQLite database: {str(e)}")
            raise

    def _init_db(self):
        """Initialize the database schema."""
        with self.lock:
            try:
                cursor = self.conn.cursor()

                # Create the main storage table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS storage_metadata (
                    key TEXT PRIMARY KEY,
                    data_hash TEXT NOT NULL,
                    compression_algo TEXT NOT NULL,
                    serialization_format TEXT NOT NULL,
                    original_size INTEGER NOT NULL,
                    compressed_size INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    storage_location TEXT NOT NULL,
                    path TEXT,
                    mmap_start INTEGER,
                    mmap_size INTEGER
                )
                ''')

                # Create indices for common queries
                cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_data_hash ON storage_metadata(data_hash)
                ''')

                cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON storage_metadata(last_accessed)
                ''')

                cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_access_count ON storage_metadata(access_count)
                ''')

                self.conn.commit()
                logger.info("Database schema initialized successfully")
            except sqlite3.Error as e:
                logger.error(f"Failed to initialize database schema: {str(e)}")
                self.conn.rollback()
                raise

    def store_metadata(self,
                       key: str,
                       memory_block: MemoryBlock) -> bool:
        """
        Store metadata for a memory block.

        Args:
            key: Key associated with the data
            memory_block: MemoryBlock containing metadata

        Returns:
            True if metadata was successfully stored
        """
        with self.lock:
            cursor = self.conn.cursor()

            try:
                cursor.execute('''
                INSERT OR REPLACE INTO storage_metadata
                (key, data_hash, compression_algo, serialization_format,
                original_size, compressed_size, created_at, last_accessed,
                access_count, storage_location, path, mmap_start, mmap_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    key,
                    memory_block.data_hash,
                    memory_block.compression_algo,
                    memory_block.serialization_format,
                    memory_block.original_size,
                    memory_block.compressed_size,
                    memory_block.created_at,
                    memory_block.last_accessed,
                    memory_block.access_count,
                    memory_block.storage_location,
                    memory_block.path,
                    memory_block.mmap_start,
                    memory_block.mmap_size
                ))

                self.conn.commit()
                return True

            except sqlite3.Error as e:
                logger.error(f"SQLite error occurred while storing metadata for key {key}: {e}")
                self.conn.rollback()
                return False

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for a key.

        Args:
            key: Key associated with the data

        Returns:
            Dictionary with metadata or None if not found
        """
        with self.lock:
            try:
                cursor = self.conn.cursor()

                cursor.execute('''
                SELECT key, data_hash, compression_algo, serialization_format,
                       original_size, compressed_size, created_at, last_accessed,
                       access_count, storage_location, path, mmap_start, mmap_size
                FROM storage_metadata
                WHERE key = ?
                ''', (key,))

                row = cursor.fetchone()

                if not row:
                    return None

                # Convert row to dictionary
                metadata = {
                    "key": row[0],
                    "data_hash": row[1],
                    "compression_algo": row[2],
                    "serialization_format": row[3],
                    "original_size": row[4],
                    "compressed_size": row[5],
                    "created_at": row[6],
                    "last_accessed": row[7],
                    "access_count": row[8],
                    "storage_location": row[9],
                    "path": row[10],
                    "mmap_start": row[11],
                    "mmap_size": row[12]
                }

                # Create a memory block from the metadata
                memory_block = MemoryBlock(
                    key=metadata["key"],
                    data_hash=metadata["data_hash"],
                    compression_algo=metadata["compression_algo"],
                    serialization_format=metadata["serialization_format"],
                    original_size=metadata["original_size"],
                    compressed_size=metadata["compressed_size"],
                    created_at=metadata["created_at"],
                    last_accessed=metadata["last_accessed"],
                    access_count=metadata["access_count"],
                    storage_location=metadata["storage_location"],
                    path=metadata["path"],
                    mmap_start=metadata["mmap_start"],
                    mmap_size=metadata["mmap_size"]
                )

                metadata["memory_block"] = memory_block
                return metadata
            except sqlite3.Error as e:
                logger.error(f"SQLite error occurred while retrieving metadata for key {key}: {e}")
                return None

    def update_access_stats(self, key: str) -> bool:
        """
        Update access statistics for a key.

        Args:
            key: Key associated with the data

        Returns:
            True if statistics were updated successfully
        """
        with self.lock:
            cursor = self.conn.cursor()
            current_time = time.time()

            try:
                cursor.execute('''
                UPDATE storage_metadata
                SET last_accessed = ?,
                    access_count = access_count + 1
                WHERE key = ?
                ''', (current_time, key))

                self.conn.commit()
                return cursor.rowcount > 0

            except sqlite3.Error as e:
                logger.error(f"SQLite error occurred while updating access stats for key {key}: {e}")
                self.conn.rollback()
                return False

    def remove_metadata(self, key: str) -> bool:
        """
        Remove metadata for a key.

        Args:
            key: Key associated with the data

        Returns:
            True if metadata was successfully removed
        """
        with self.lock:
            cursor = self.conn.cursor()

            try:
                cursor.execute('''
                DELETE FROM storage_metadata
                WHERE key = ?
                ''', (key,))

                self.conn.commit()
                return cursor.rowcount > 0

            except sqlite3.Error as e:
                logger.error(f"SQLite error occurred while removing metadata for key {key}: {e}")
                self.conn.rollback()
                return False

    def get_least_accessed_keys(self, count: int = 10) -> List[str]:
        """
        Get the least accessed keys, for eviction policies.

        Args:
            count: Number of keys to retrieve

        Returns:
            List of keys sorted by access count and last accessed time
        """
        with self.lock:
            try:
                cursor = self.conn.cursor()

                cursor.execute('''
                SELECT key
                FROM storage_metadata
                ORDER BY access_count ASC, last_accessed ASC
                LIMIT ?
                ''', (count,))

                return [row[0] for row in cursor.fetchall()]
            except sqlite3.Error as e:
                logger.error(f"SQLite error occurred while retrieving least accessed keys: {e}")
                return []

    def get_duplicate_data_hashes(self) -> List[str]:
        """
        Get data hashes that appear multiple times (duplicates).

        Returns:
            List of data hashes that have duplicates
        """
        with self.lock:
            try:
                cursor = self.conn.cursor()

                cursor.execute('''
                SELECT data_hash
                FROM storage_metadata
                GROUP BY data_hash
                HAVING COUNT(*) > 1
                ''')

                return [row[0] for row in cursor.fetchall()]
            except sqlite3.Error as e:
                logger.error(f"SQLite error occurred while retrieving duplicate data hashes: {e}")
                return []

    def get_keys_by_data_hash(self, data_hash: str) -> List[str]:
        """
        Get all keys that have the same data hash.

        Args:
            data_hash: The data hash to look for

        Returns:
            List of keys with the specified data hash
        """
        with self.lock:
            try:
                cursor = self.conn.cursor()

                cursor.execute('''
                SELECT key
                FROM storage_metadata
                WHERE data_hash = ?
                ''', (data_hash,))

                return [row[0] for row in cursor.fetchall()]
            except sqlite3.Error as e:
                logger.error(f"SQLite error occurred while retrieving keys by data hash {data_hash}: {e}")
                return []

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the stored data.

        Returns:
            Dictionary with statistics
        """
        with self.lock:
            try:
                cursor = self.conn.cursor()
                stats = {}

                # Total items
                cursor.execute('SELECT COUNT(*) FROM storage_metadata')
                stats['total_items'] = cursor.fetchone()[0]

                # Total original size
                cursor.execute('SELECT SUM(original_size) FROM storage_metadata')
                stats['total_original_size'] = cursor.fetchone()[0] or 0

                # Total compressed size
                cursor.execute('SELECT SUM(compressed_size) FROM storage_metadata')
                stats['total_compressed_size'] = cursor.fetchone()[0] or 0

                # Compression ratio
                if stats['total_original_size'] > 0:
                    stats['compression_ratio'] = stats['total_original_size'] / stats['total_compressed_size']
                else:
                    stats['compression_ratio'] = 5.0

                # Items per storage location
                cursor.execute('''
                SELECT storage_location, COUNT(*), SUM(compressed_size)
                FROM storage_metadata
                GROUP BY storage_location
                ''')

                stats['storage_distribution'] = {}
                for row in cursor.fetchall():
                    location, count, size = row
                    stats['storage_distribution'][location] = {
                        'count': count,
                        'size': size or 0
                    }

                # Items per compression algorithm
                cursor.execute('''
                SELECT compression_algo, COUNT(*), SUM(original_size), SUM(compressed_size)
                FROM storage_metadata
                GROUP BY compression_algo
                ''')

                stats['compression_algorithms'] = {}
                for row in cursor.fetchall():
                    algo, count, orig_size, comp_size = row
                    ratio = orig_size / comp_size if comp_size and orig_size else 1.0
                    stats['compression_algorithms'][algo] = {
                        'count': count,
                        'original_size': orig_size or 0,
                        'compressed_size': comp_size or 0,
                        'ratio': ratio
                    }

                # Items per serialization format
                cursor.execute('''
                SELECT serialization_format, COUNT(*)
                FROM storage_metadata
                GROUP BY serialization_format
                ''')

                stats['serialization_formats'] = {}
                for row in cursor.fetchall():
                    format_name, count = row
                    stats['serialization_formats'][format_name] = count

                # Age statistics
                current_time = time.time()
                cursor.execute('''
                SELECT
                    COUNT(*) FILTER (WHERE created_at > ?),
                    COUNT(*) FILTER (WHERE created_at <= ? AND created_at > ?),
                    COUNT(*) FILTER (WHERE created_at <= ?)
                FROM storage_metadata
                ''', (
                    current_time - 3600,  # Last hour
                    current_time - 3600,  # More than hour old
                    current_time - 86400,  # Between 1 hour and 1 day
                    current_time - 86400   # More than 1 day old
                ))

                last_hour, between_hour_day, older_than_day = cursor.fetchone()
                stats['age_distribution'] = {
                    'last_hour': last_hour or 0,
                    'between_hour_day': between_hour_day or 0,
                    'older_than_day': older_than_day or 0
                }

                return stats
            except sqlite3.Error as e:
                logger.error(f"SQLite error occurred while retrieving stats: {e}")
                return {"error": str(e)}

    def cleanup(self):
        """Close database connection"""
        with self.lock:
            if self.conn:
                try:
                    self.conn.close()
                    self.conn = None
                    logger.info("Database connection closed")
                except sqlite3.Error as e:
                    logger.error(f"Error closing database connection: {e}")


class AdvancedMemoryManager:
    """Advanced memory manager with tiered storage and optimization features"""

    def __init__(self, memory_limit_mb: int = 1024, disk_path: str = "j:\\MyStorage5"):
        """
        Initialize the Advanced Memory Manager.
        
        Args:
            memory_limit_mb: Memory limit in megabytes
            disk_path: Path to store data on disk
        """
        logger.info(f"Initializing AdvancedMemoryManager with memory limit {memory_limit_mb}MB and disk path {disk_path}")
        
        self.storage = StorageManager(disk_path=disk_path, memory_limit=memory_limit_mb * 1024 * 1024)
        self.metadata = MetadataManager(os.path.join(disk_path, "metadata.db"))
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.RLock()

        # Deduplication cache - maps data hash to key
        self.dedup_cache = {}

        # LRU cache for frequently accessed data
        self.lru_cache = OrderedDict()
        self.lru_cache_size = 100 * 1024 * 1024  # 100MB default LRU cache size
        self.current_lru_cache_usage = 100

        # Start background optimization thread
        self.should_stop = False
        self.bg_thread = threading.Thread(target=self._background_optimizer)
        self.bg_thread.daemon = True
        self.bg_thread.start()
        
        logger.info("AdvancedMemoryManager initialized successfully")

    def _compute_hash(self, data: bytes) -> str:
        """Compute a hash of the data for deduplication"""
        return hashlib.sha256(data).hexdigest()

    def _select_storage_location(self, data_size: int, importance: float = 0.5) -> str:
        """
        Select appropriate storage location based on size and importance

        Args:
            data_size: Size of the data in bytes
            importance: Importance score (0-1) with higher values prioritizing memory storage

        Returns:
            Storage location: 'memory', 'disk', or 'hybrid'
        """
        stats = self.storage.get_storage_stats()

        # If data is small or very important, try to keep in memory
        if (data_size < 10240 or importance > 0.8) and \
           (stats['memory']['usage'] + data_size <= stats['memory']['limit']):
            return 'memory'

        # For medium-sized data with medium importance, use hybrid storage
        if data_size < 1024 * 1024 and importance > 0.3:
            return 'hybrid'

        # For large or less important data, use disk
        return 'disk'

    def _evict_data_if_needed(self, needed_space: int):
        """Evict least recently used data if memory is running low"""
        with self.lock:
            stats = self.storage.get_storage_stats()

            # Check if we need to free up memory
            if stats['memory']['usage'] + needed_space <= stats['memory']['limit']:
                return

            # Calculate how much space we need to free
            space_to_free = (stats['memory']['usage'] + needed_space) - stats['memory']['limit']
            logger.info(f"Need to free {space_to_free} bytes of memory")

            # Get candidates for eviction
            candidates = self.metadata.get_least_accessed_keys(20)

            freed_space = 0
            for key in candidates:
                block = self.metadata.get_metadata(key)
                if not block or block['storage_location'] != 'memory':
                    continue

                try:
                    # Retrieve data
                    data = self.storage.retrieve_from_memory(key)
                    if data:
                        # Move data to disk
                        new_path, _ = self.storage.store_on_disk(key, data)
                        freed_space += len(data)
                        self.storage.remove_data(key, 'memory')

                        # Update metadata
                        block['memory_block'].storage_location = 'disk'
                        block['memory_block'].path = new_path
                        self.metadata.store_metadata(key, block['memory_block'])
                        
                        logger.info(f"Evicted key {key} to disk, freed {len(data)} bytes")
                except Exception as e:
                    logger.error(f"Error evicting data for key {key}: {str(e)}")
                    continue

                if freed_space >= space_to_free:
                    break
                    
            logger.info(f"Eviction complete, freed {freed_space} bytes")

    def _evict_lru_cache_if_needed(self, needed_space: int):
        """Evict least recently used data from LRU cache if needed"""
        with self.lock:
            if self.current_lru_cache_usage + needed_space <= self.lru_cache_size:
                return

            space_to_free = (self.current_lru_cache_usage + needed_space) - self.lru_cache_size
            logger.info(f"Need to free {space_to_free} bytes from LRU cache")

            freed_space = 0
            while self.lru_cache and freed_space < space_to_free:
                key, cache_entry = self.lru_cache.popitem(last=False)
                freed_space += cache_entry['size']
                logger.debug(f"Evicted key {key} from LRU cache, freed {cache_entry['size']} bytes")

            self.current_lru_cache_usage -= freed_space
            logger.info(f"LRU cache eviction complete, freed {freed_space} bytes")

    def _background_optimizer(self):
        """Background thread for optimizing storage with improved thread safety"""
        logger.info("Background optimizer thread started")
        
        while not self.should_stop:
            try:
                # Sleep between optimization runs
                time.sleep(60)
                
                if self.should_stop:
                    break
                    
                logger.info("Running background optimization")

                # Find duplicate data
                with self.lock:
                    duplicate_hashes = self.metadata.get_duplicate_data_hashes()
                    logger.info(f"Found {len(duplicate_hashes)} duplicate hashes")

                for data_hash in duplicate_hashes:
                    # Skip if we're shutting down
                    if self.should_stop:
                        break
                        
                    with self.lock:
                        keys = self.metadata.get_keys_by_data_hash(data_hash)
                        if len(keys) <= 1:
                            continue

                        # Check if any are in memory
                        main_key = None
                        main_location = None

                        for key in keys:
                            block = self.metadata.get_metadata(key)
                            if block and block['storage_location'] == 'memory':
                                main_key = key
                                main_location = 'memory'
                                break

                        # If none in memory, try hybrid, then disk
                        if not main_key:
                            for key in keys:
                                block = self.metadata.get_metadata(key)
                                if block and block['storage_location'] == 'hybrid':
                                    main_key = key
                                    main_location = 'hybrid'
                                    break

                        if not main_key and keys:
                            main_key = keys[0]
                            block = self.metadata.get_metadata(main_key)
                            if block:
                                main_location = block['storage_location']

                        # Update deduplication cache
                        if main_key:
                            self.dedup_cache[data_hash] = main_key
                            logger.debug(f"Updated deduplication cache for hash {data_hash} to key {main_key}")

                # Run periodic disk cleanup
                if not self.should_stop:
                    self._cleanup_disk_space()

                # Optimize memory usage
                if not self.should_stop:
                    self._optimize_memory_usage()
                    
                logger.info("Background optimization complete")

            except Exception as e:
                logger.error(f"Error in background optimizer: {e}")
                logger.error(traceback.format_exc())
                # Sleep a bit to avoid tight error loops
                time.sleep(5)

    @lru_cache(maxsize=1000)
    def _compute_hash_cached(self, data: bytes) -> str:
        """Cached version of hash computation for performance"""
        return self._compute_hash(data)

    def _handle_deduplication(self, key: str, data_hash: str, serialized: bytes, original_size: int) -> Dict[str, Any]:
        """Handle deduplication of data"""
        with self.lock:
            dedup_key = self.dedup_cache.get(data_hash)
            if dedup_key and dedup_key != key:
                dedup_block = self.metadata.get_metadata(dedup_key)
                if dedup_block:
                    block = MemoryBlock(
                        key=key,
                        data_hash=data_hash,
                        compression_algo=dedup_block['compression_algo'],
                        serialization_format=dedup_block['serialization_format'],
                        original_size=original_size,
                        compressed_size=dedup_block['compressed_size'],
                        created_at=time.time(),
                        last_accessed=time.time(),
                        access_count=0,
                        storage_location=dedup_block['storage_location'],
                        path=dedup_block['path'],
                        mmap_start=dedup_block['mmap_start'],
                        mmap_size=dedup_block['mmap_size']
                    )
                    self.metadata.store_metadata(key, block)
                    logger.info(f"Deduplicated key {key} to reference {dedup_key}")
                    return {
                        "status": "deduped",
                        "key": key,
                        "reference": dedup_key,
                        "original_size": original_size,
                        "storage_location": dedup_block['storage_location']
                    }
        return {"status": "new", "key": key}

    def _store_data_in_location(self, key: str, compressed: bytes, location: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        """Store data in the specified location"""
        try:
            if location == 'memory':
                size = self.storage.store_in_memory(key, compressed)
                return None, None, size
            elif location == 'disk':
                path, size = self.storage.store_on_disk(key, compressed)
                return path, None, size
            elif location == 'hybrid':
                path, mmap_start, mmap_size = self.storage.store_in_hybrid(key, compressed)
                return path, mmap_start, mmap_size
            else:
                raise ValueError(f"Unknown storage location: {location}")
        except Exception as e:
            logger.error(f"Error storing data for key {key} in location {location}: {str(e)}")
            # Fallback to disk storage
            try:
                path, size = self.storage.store_on_disk(key, compressed)
                return path, None, size
            except Exception as e2:
                logger.error(f"Fallback to disk storage also failed: {str(e2)}")
                raise ValueError(f"Failed to store data in any storage location: {str(e)}, then {str(e2)}")

    def _update_lru_cache(self, key: str, data: bytes, size: int):
        """Update the LRU cache with the new data"""
        with self.lock:
            # Remove if already in cache
            if key in self.lru_cache:
                old_size = self.lru_cache[key]['size']
                self.current_lru_cache_usage -= old_size
                del self.lru_cache[key]
                
            # Add to cache if there's enough space
            self._evict_lru_cache_if_needed(size)
            self.lru_cache[key] = {
                'data': data,
                'size': size,
                'last_accessed': time.time()
            }
            self.current_lru_cache_usage += size
            logger.debug(f"Updated LRU cache for key {key}, size {size} bytes")

    async def store_data(self, key: str, data: Any, importance: float = 0.5,
                         compression_algo: str = "auto", serialization_format: str = "auto",
                         override_existing: bool = True) -> Dict[str, Any]:
        """
        Store data with advanced features asynchronously

        Args:
            key: The key to store data under
            data: The data to store
            importance: Importance score (0-1) affecting storage location
            compression_algo: Compression algorithm to use ('auto' or one of CompressionEngine.ALGORITHMS)
            serialization_format: Serialization format or "auto"
            override_existing: Whether to override if key exists

        Returns:
            Dictionary with storage information
        """
        try:
            with self.lock:
                if not override_existing and self.metadata.get_metadata(key):
                    logger.info(f"Key {key} already exists and override_existing=False")
                    return {"status": "exists", "key": key}

                # Serialize the data
                logger.info(f"Serializing data for key {key} with format {serialization_format}")
                serialized, serial_format, _, _ = Serializer.serialize(data, serialization_format)
                original_size = len(serialized)
                data_hash = self._compute_hash_cached(serialized)

                # Check for deduplication
                if data_hash in self.dedup_cache and not override_existing:
                    dedup_result = self._handle_deduplication(key, data_hash, serialized, original_size)
                    if dedup_result["status"] == "deduped":
                        return dedup_result

                # Compress the data
                logger.info(f"Compressing data for key {key} with algorithm {compression_algo}")
                compressed, comp_algo, comp_ratio = compress_data(serialized, compression_algo)
                compressed_size = len(compressed)

                # Select storage location
                location = self._select_storage_location(compressed_size, importance)
                logger.info(f"Selected storage location {location} for key {key}")
                
                # Ensure enough space in memory if needed
                if location == 'memory':
                    self._evict_data_if_needed(compressed_size)

                # Store the data
                path, mmap_start, mmap_size = self._store_data_in_location(key, compressed, location)

                # Create and store metadata
                block = MemoryBlock(
                    key=key,
                    data_hash=data_hash,
                    compression_algo=comp_algo,
                    serialization_format=serial_format,
                    original_size=original_size,
                    compressed_size=compressed_size,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    access_count=0,
                    storage_location=location,
                    path=path,
                    mmap_start=mmap_start,
                    mmap_size=mmap_size
                )

                self.metadata.store_metadata(key, block)
                self.dedup_cache[data_hash] = key

                # Update LRU cache if stored in memory
                if location == 'memory':
                    self._update_lru_cache(key, compressed, compressed_size)

                logger.info(f"Successfully stored data for key {key}, size {original_size} bytes, compressed to {compressed_size} bytes")
                return {
                    "status": "stored",
                    "key": key,
                    "original_size": original_size,
                    "compressed_size": compressed_size,
                    "compression_ratio": comp_ratio,
                    "storage_location": location,
                    "compression_algorithm": comp_algo,
                    "serialization_format": serial_format
                }
        except Exception as e:
            logger.error(f"Error storing data for key {key}: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Failed to store data: {str(e)}")

    async def retrieve_data(self, key: str) -> Optional[Any]:
        """Retrieve data by key asynchronously"""
        try:
            with self.lock:
                # Check LRU cache first
                if key in self.lru_cache:
                    logger.debug(f"Retrieved key {key} from LRU cache")
                    self.lru_cache.move_to_end(key)
                    self.lru_cache[key]['last_accessed'] = time.time()
                    compressed_data = self.lru_cache[key]['data']
                    
                    # Get metadata for decompression
                    block = self.metadata.get_metadata(key)
                    if not block:
                        logger.warning(f"Found key {key} in LRU cache but not in metadata")
                        return None
                else:
                    # Get metadata
                    block = self.metadata.get_metadata(key)
                    if not block:
                        logger.warning(f"Key {key} not found in metadata")
                        return None

                    # Update access stats
                    self.metadata.update_access_stats(key)

                    # Retrieve compressed data based on storage location
                    logger.info(f"Retrieving data for key {key} from {block['storage_location']}")
                    compressed_data = None
                    if block['storage_location'] == 'memory':
                        compressed_data = self.storage.retrieve_from_memory(key)
                    elif block['storage_location'] == 'disk':
                        compressed_data = self.storage.retrieve_from_disk(block['path'])
                    elif block['storage_location'] == 'hybrid':
                        compressed_data = self.storage.retrieve_from_hybrid(block['path'], block['mmap_start'], block['mmap_size'])

                    if not compressed_data:
                        logger.warning(f"Failed to retrieve data for key {key}")
                        return None

                    # Update LRU cache
                    self._update_lru_cache(key, compressed_data, block['compressed_size'])

                # Decompress and deserialize
                try:
                    raw_data = CompressionEngine.decompress(compressed_data, block['compression_algo'], block['original_size'])
                    result = Serializer.deserialize(raw_data, block['serialization_format'], trusted=True)
                    logger.info(f"Successfully retrieved and deserialized data for key {key}")
                    return result
                except Exception as e:
                    logger.error(f"Error decompressing or deserializing data for key {key}: {str(e)}")
                    return None
        except Exception as e:
            logger.error(f"Error retrieving data for key {key}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def remove_data(self, key: str) -> bool:
        """Remove data and metadata by key asynchronously"""
        try:
            with self.lock:
                block = self.metadata.get_metadata(key)
                if not block:
                    logger.warning(f"Key {key} not found for removal")
                    return False

                logger.info(f"Removing data for key {key} from {block['storage_location']}")
                
                # Remove from LRU cache if present
                if key in self.lru_cache:
                    self.current_lru_cache_usage -= self.lru_cache[key]['size']
                    del self.lru_cache[key]

                # Remove from storage
                self.storage.remove_data(key, block['storage_location'], block['path'],
                                        (block['mmap_start'], block['mmap_size']) if block['mmap_start'] is not None else None)

                # Remove metadata
                self.metadata.remove_metadata(key)

                # Remove from deduplication cache if it's there
                for h, k in list(self.dedup_cache.items()):
                    if k == key:
                        del self.dedup_cache[h]

                logger.info(f"Successfully removed data for key {key}")
                return True
        except Exception as e:
            logger.error(f"Error removing data for key {key}: {str(e)}")
            return False

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get detailed storage statistics"""
        try:
            with self.lock:
                storage_stats = self.storage.get_storage_stats()
                metadata_stats = self.metadata.get_stats()

                system_memory = psutil.virtual_memory()

                stats = {
                    "storage": storage_stats,
                    "metadata": metadata_stats,
                    "system": {
                        "total_memory": system_memory.total,
                        "available_memory": system_memory.available,
                        "memory_percent": system_memory.percent
                    },
                    "deduplication": {
                        "cached_hashes": len(self.dedup_cache)
                    },
                    "lru_cache": {
                        "usage": self.current_lru_cache_usage,
                        "limit": self.lru_cache_size,
                        "usage_pct": (self.current_lru_cache_usage / self.lru_cache_size) * 100 if self.lru_cache_size > 0 else 0,
                        "items": len(self.lru_cache)
                    }
                }

                return stats
        except Exception as e:
            logger.error(f"Error getting storage stats: {str(e)}")
            return {"error": str(e)}

    def _cleanup_disk_space(self) -> Dict[str, Any]:
        """Remove unused or obsolete files from disk"""
        deleted_files = 0
        freed_space = 0

        try:
            logger.info("Running disk space cleanup")
            
            # Get all files in storage directory
            for filename in os.listdir(self.storage.disk_path):
                filepath = os.path.join(self.storage.disk_path, filename)
                if os.path.isfile(filepath) and filepath != str(self.storage.mmap_file_path):
                    # Check if the file is referenced in metadata
                    with self.lock:
                        cursor = self.metadata.conn.cursor()
                        cursor.execute('SELECT COUNT(*) FROM storage_metadata WHERE path = ?', (filepath,))
                        count = cursor.fetchone()[0]
                        
                    if count == 0:
                        try:
                            file_size = os.path.getsize(filepath)
                            os.remove(filepath)
                            deleted_files += 1
                            freed_space += file_size
                            logger.info(f"Deleted unreferenced file {filepath}, freed {file_size} bytes")
                        except OSError as e:
                            logger.error(f"Failed to delete file {filepath}: {str(e)}")

            logger.info(f"Disk cleanup complete: deleted {deleted_files} files, freed {freed_space} bytes")
            return {
                "deleted_files": deleted_files,
                "freed_space": freed_space
            }
        except Exception as e:
            logger.error(f"Error during disk cleanup: {str(e)}")
            return {
                "error": str(e),
                "deleted_files": deleted_files,
                "freed_space": freed_space
            }

    def _optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage by moving data to disk"""
        try:
            with self.lock:
                stats = self.storage.get_storage_stats()
                memory_usage = stats['memory']['usage']
                memory_limit = stats['memory']['limit']

                if memory_usage < memory_limit * 0.6:  # Only optimize if memory usage is above 60%
                    logger.info("Memory usage below 60% threshold, skipping optimization")
                    return {
                        "status": "not_optimized",
                        "reason": "Memory usage below 60% of limit"
                    }

                logger.info(f"Optimizing memory usage: current {memory_usage} bytes, limit {memory_limit} bytes")
                
                # Get candidates for eviction
                candidates = self.metadata.get_least_accessed_keys(50)

                moved_data = 0
                moved_items = 0
                for key in candidates:
                    block = self.metadata.get_metadata(key)
                    if not block or block['storage_location'] != 'memory':
                        continue

                    try:
                        # Retrieve data
                        data = self.storage.retrieve_from_memory(key)
                        if data:
                            # Move data to disk
                            new_path, _ = self.storage.store_on_disk(key, data)
                            data_size = len(data)
                            self.storage.remove_data(key, 'memory')
                            moved_data += data_size
                            moved_items += 1

                            # Update metadata
                            block['memory_block'].storage_location = 'disk'
                            block['memory_block'].path = new_path
                            block['memory_block'].mmap_start = None
                            block['memory_block'].mmap_size = None
                            self.metadata.store_metadata(key, block['memory_block'])
                            
                            logger.info(f"Moved key {key} from memory to disk, size {data_size} bytes")
                    except Exception as e:
                        logger.error(f"Error moving data for key {key} to disk: {str(e)}")
                        continue

                    # Stop if we've moved enough data
                    if moved_data >= memory_limit * 0.1:  # Move up to 10% of memory limit
                        break

                logger.info(f"Memory optimization complete: moved {moved_items} items, {moved_data} bytes to disk")
                return {
                    "status": "optimized",
                    "moved_data": moved_data,
                    "moved_items": moved_items
                }
        except Exception as e:
            logger.error(f"Error optimizing memory usage: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def optimize(self) -> Dict[str, Any]:
        """Run optimization process and return results"""
        try:
            logger.info("Running manual optimization")
            results = {"steps": []}

            # 1. Find and consolidate duplicates
            duplicate_hashes = self.metadata.get_duplicate_data_hashes()
            deduped_count = 0
            saved_space = 0

            for data_hash in duplicate_hashes:
                with self.lock:
                    keys = self.metadata.get_keys_by_data_hash(data_hash)
                    if len(keys) <= 1:
                        continue

                    # Pick one key to keep data
                    main_key = keys[0]
                    main_block = self.metadata.get_metadata(main_key)
                    if not main_block:
                        continue

                    # Make other keys reference the main key
                    for other_key in keys[1:]:
                        other_block = self.metadata.get_metadata(other_key)
                        if not other_block:
                            continue

                        # Skip if already properly referenced
                        if other_block['storage_location'] == main_block['storage_location'] and \
                        other_block['path'] == main_block['path']:
                            continue

                        # Update metadata to point to main key's data
                        other_block['memory_block'].storage_location = main_block['storage_location']
                        other_block['memory_block'].path = main_block['path']
                        other_block['memory_block'].compression_algo = main_block['compression_algo']
                        other_block['memory_block'].compressed_size = main_block['compressed_size']
                        other_block['memory_block'].mmap_start = main_block['mmap_start']
                        other_block['memory_block'].mmap_size = main_block['mmap_size']

                        # Remove redundant storage
                        if other_block['storage_location'] == 'disk' and other_block['path']:
                            try:
                                os.remove(other_block['path'])
                            except OSError:
                                pass

                        # Update metadata
                        self.metadata.store_metadata(other_key, other_block['memory_block'])

                        deduped_count += 1
                        saved_space += other_block['original_size']
                        logger.info(f"Deduplicated key {other_key} to reference {main_key}")

            # 2. Cleanup unused disk space
            disk_cleanup_result = self._cleanup_disk_space()

            # 3. Optimize memory usage
            memory_optimization_result = self._optimize_memory_usage()

            results["steps"].append({
                "deduplication": {
                    "duplicates_removed": deduped_count,
                    "space_saved": saved_space
                },
                "disk_cleanup": disk_cleanup_result,
                "memory_optimization": memory_optimization_result
            })

            logger.info(f"Manual optimization complete: deduplicated {deduped_count} items, saved {saved_space} bytes")
            return results
        except Exception as e:
            logger.error(f"Error during manual optimization: {str(e)}")
            return {"error": str(e)}

    def shutdown(self):
        """Properly shut down the memory manager."""
        logger.info("Shutting down Advanced Memory Manager...")
        
        # Signal the background thread to stop
        self.should_stop = True
        
        # Wait for background thread to finish (with timeout)
        if hasattr(self, 'bg_thread') and self.bg_thread.is_alive():
            self.bg_thread.join(timeout=5.0)
            if self.bg_thread.is_alive():
                logger.warning("Background thread did not terminate gracefully")
        
        # Shutdown thread pool
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        
        # Clean up storage
        if hasattr(self, 'storage'):
            self.storage.cleanup()
        
        # Clean up metadata
        if hasattr(self, 'metadata'):
            self.metadata.cleanup()
        
        logger.info("Advanced Memory Manager shutdown complete")


def test_compression_algorithms():
    """Test all compression algorithms to verify they work correctly."""
    print("Testing compression algorithms...")
    
    # Create test data
    test_data = b"This is a test string that will be compressed." * 100
    
    # Test each algorithm
    for algorithm in CompressionEngine.ALGORITHMS:
        try:
            print(f"Testing {algorithm}...")
            compressed, original_size = CompressionEngine.compress(test_data, algorithm)
            decompressed = CompressionEngine.decompress(compressed, algorithm, original_size)
            
            # Verify the decompressed data matches the original
            if decompressed == test_data:
                compression_ratio = len(test_data) / len(compressed)
                print(f"  ✓ {algorithm}: Success! Ratio: {compression_ratio:.2f}x")
            else:
                print(f"  ✗ {algorithm}: Data mismatch after decompression")
        except Exception as e:
            print(f"  ✗ {algorithm}: Error - {str(e)}")
    
    print("Compression algorithm testing complete")


def benchmark_compression_algorithms():
    """Benchmark all compression algorithms for speed and ratio."""
    print("Benchmarking compression algorithms...")
    
    # Create different types of test data
    test_datasets = {
        "text": b"This is a sample text that will be compressed. " * 1000,
        "binary": os.urandom(100000),
        "json": json.dumps({"data": [i for i in range(1000)]}).encode() * 10,
        "zeros": b"\0" * 100000
    }
    
    results = {}
    
    for data_type, data in test_datasets.items():
        print(f"\nTesting with {data_type} data ({len(data)} bytes):")
        results[data_type] = {}
        
        for algorithm in CompressionEngine.ALGORITHMS:
            try:
                # Measure compression speed
                start_time = time.time()
                compressed, _ = CompressionEngine.compress(data, algorithm)
                compression_time = time.time() - start_time
                
                # Measure decompression speed
                start_time = time.time()
                decompressed = CompressionEngine.decompress(compressed, algorithm)
                decompression_time = time.time() - start_time
                
                # Calculate compression ratio
                compression_ratio = len(data) / len(compressed)
                
                # Verify correctness
                if decompressed == data:
                    print(f"  ✓ {algorithm}: Ratio: {compression_ratio:.2f}x, Compression: {compression_time:.4f}s, Decompression: {decompression_time:.4f}s")
                    results[data_type][algorithm] = {
                        "ratio": compression_ratio,
                        "compression_time": compression_time,
                        "decompression_time": decompression_time,
                        "compressed_size": len(compressed)
                    }
                else:
                    print(f"  ✗ {algorithm}: Data mismatch after decompression")
            except Exception as e:
                print(f"  ✗ {algorithm}: Error - {str(e)}")
    
    # Print summary
    print("\nCompression Benchmark Summary:")
    for data_type in results:
        print(f"\n{data_type.upper()} DATA:")
        print(f"{'Algorithm':<10} {'Ratio':<10} {'Comp Time':<15} {'Decomp Time':<15} {'Size'}")
        print("-" * 60)
        
        for algorithm in sorted(results[data_type], key=lambda x: results[data_type][x]["ratio"], reverse=True):
            result = results[data_type][algorithm]
            print(f"{algorithm:<10} {result['ratio']:<10.2f} {result['compression_time']:<15.4f} {result['decompression_time']:<15.4f} {result['compressed_size']}")
    
    print("\nBenchmark complete")


# Example usage:
async def main():
    manager = AdvancedMemoryManager(memory_limit_mb=256, disk_path="j:\\MyStorage5")

    try:
        # Store data
        data_to_store = {
            "name": "Example",
            "value": 42,
            "array": np.array([1, 2, 3, 4, 5])
        }
        result = await manager.store_data("example_key", data_to_store)
        print("Store Result:", result)

        # Retrieve data
        retrieved_data = await manager.retrieve_data("example_key")
        print("Retrieved Data:", retrieved_data)

        # Remove data
        removed = await manager.remove_data("example_key")
        print("Remove Result:", removed)

        # Get storage stats
        stats = manager.get_storage_stats()
        print("Storage Stats:", stats)

        # Run optimization
        optimization_result = manager.optimize()
        print("Optimization Result:", optimization_result)
        
        # Test compression algorithms
        test_compression_algorithms()
        
        # Properly shut down
        manager.shutdown()
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()
        manager.shutdown()


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Uncomment to run the benchmark
    # benchmark_compression_algorithms()