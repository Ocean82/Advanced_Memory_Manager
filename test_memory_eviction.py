import asyncio
import numpy as np
import os
import time
import sys
import traceback

# Add the parent directory to the path so we can import the module
sys.path.append('.')  # This allows importing from the current directory

# Import the AdvancedMemoryManager from your file
from storage50 import AdvancedMemoryManager

async def test_memory_eviction():
    # Use a VERY small memory limit (1MB)
    memory_limit_mb = 1
    
    # Create a clean test directory
    test_dir = "j:/MyStorage5/test_eviction"
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"Starting memory eviction test with {memory_limit_mb}MB limit")
    manager = AdvancedMemoryManager(memory_limit_mb=memory_limit_mb, disk_path=test_dir)
    
    # Override the storage location selection to ALWAYS use memory
    original_select = manager._select_storage_location
    manager._select_storage_location = lambda size, importance: 'memory'
    
    try:
        print("Phase 1: Filling memory until eviction occurs...")
        for i in range(20):
            # Create data that's about 200KB each
            array_data = np.random.rand(25000)  # 25000 * 8 bytes = 200KB
            data = {
                "id": f"object_{i}",
                "data": array_data
            }
            
            result = await manager.store_data(f"key_{i}", data, importance=0.9)
            print(f"Stored object {i}: {result['compressed_size']/1024:.1f}KB in {result['storage_location']}")
            
            # Print memory stats after each object
            stats = manager.get_storage_stats()
            memory_usage = stats['storage']['memory']['usage']
            memory_limit = stats['storage']['memory']['limit']
            memory_pct = stats['storage']['memory']['usage_pct']
            print(f"Memory usage: {memory_usage/1024/1024:.2f}MB / {memory_limit/1024/1024:.2f}MB ({memory_pct:.1f}%)")
            
            # Check storage locations of all keys
            if i >= 5:
                memory_count = 0
                disk_count = 0
                hybrid_count = 0
                
                for j in range(i+1):
                    key = f"key_{j}"
                    metadata = manager.metadata.get_metadata(key)
                    if metadata:
                        location = metadata['storage_location']
                        if location == 'memory':
                            memory_count += 1
                        elif location == 'disk':
                            disk_count += 1
                        elif location == 'hybrid':
                            hybrid_count += 1
                
                print(f"Storage distribution: memory={memory_count}, disk={disk_count}, hybrid={hybrid_count}")
                
                # If we see data has moved to disk, we've confirmed eviction works
                if disk_count > 0:
                    print("SUCCESS: Eviction has occurred! Some data moved to disk.")
        
        # Restore original method
        manager._select_storage_location = original_select
        
        # Clean up
        manager.shutdown()
        
    except Exception as e:
        print(f"Error in test: {str(e)}")
        traceback.print_exc()
        manager.shutdown()

if __name__ == "__main__":
    asyncio.run(test_memory_eviction())