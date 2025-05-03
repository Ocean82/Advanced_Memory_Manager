import asyncio
import numpy as np
import random
import os
import time
import logging
from advanced_memory_manager import AdvancedMemoryManager

async def test_memory_pressure():
    # Use a small memory limit to force eviction (5MB)
    memory_limit_mb = 5
    
    # Create a clean test directory
    test_dir = "test_storage"
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"Starting memory pressure test with {memory_limit_mb}MB limit")
    manager = AdvancedMemoryManager(memory_limit_mb=memory_limit_mb, disk_path=test_dir)
    
    try:
        # Store objects directly in memory by forcing the storage location
        stored_keys = []
        
        # Override the _select_storage_location method to force memory storage
        original_select = manager._select_storage_location
        manager._select_storage_location = lambda size, importance: 'memory'
        
        print("Phase 1: Storing objects directly in memory...")
        for i in range(10):
            # Create data that's about 0.5MB each
            array_data = np.random.rand(62500)  # 62500 * 8 bytes = 500KB
            data = {
                "id": f"object_{i}",
                "data": array_data
            }
            
            result = await manager.store_data(f"key_{i}", data, importance=0.9)
            stored_keys.append(f"key_{i}")
            print(f"Stored object {i}: {result['compressed_size']/1024:.1f}KB in {result['storage_location']}")
            
            # Print memory stats after each object
            stats = manager.get_storage_stats()
            memory_usage = stats['storage']['memory']['usage']
            memory_limit = stats['storage']['memory']['limit']
            memory_pct = stats['storage']['memory']['usage_pct']
            print(f"Memory usage: {memory_usage/1024/1024:.2f}MB / {memory_limit/1024/1024:.2f}MB ({memory_pct:.1f}%)")
        
        # Restore original method
        manager._select_storage_location = original_select
        
        # Phase 2: Verify data retrieval and storage locations
        print("\nPhase 2: Verifying data retrieval and storage locations...")
        for i in range(10):
            key = f"key_{i}"
            data = await manager.retrieve_data(key)
            if data:
                # Get metadata to check storage location
                metadata = manager.metadata.get_metadata(key)
                location = metadata['storage_location'] if metadata else "unknown"
                print(f"Retrieved {key}: id={data['id']}, storage={location}")
            else:
                print(f"Failed to retrieve {key}")
        
        # Phase 3: Force optimization
        print("\nPhase 3: Running optimization...")
        optimization_result = manager.optimize()
        print(f"Optimization completed")
        
        # Check storage distribution after optimization
        stats = manager.get_storage_stats()
        print("\nStorage distribution after optimization:")
        for location, info in stats['storage'].items():
            if isinstance(info, dict) and 'usage' in info:
                print(f"  {location}: {info['usage']/1024/1024:.2f}MB ({info.get('items', 0)} items)")
        
        # Clean up
        manager.shutdown()
        
    except Exception as e:
        print(f"Error in test: {str(e)}")
        import traceback
        traceback.print_exc()
        manager.shutdown()

if __name__ == "__main__":
    asyncio.run(test_memory_pressure())