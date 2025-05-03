async def main():
    # Use a small memory limit to force eviction
    memory_limit_mb = 10
    manager = AdvancedMemoryManager(memory_limit_mb=memory_limit_mb, disk_path=".cache/storage")
    
    try:
        print("Storing large objects to fill memory...")
        
        # Store 15 objects of ~1MB each
        for i in range(15):
            # Create a large array (~1MB)
            array_data = np.random.rand(125000)  # ~1MB of float64 data
            data = {
                "id": f"object_{i}",
                "data": array_data
            }
            
            # Store with medium importance
            result = await manager.store_data(f"key_{i}", data, importance=0.5)
            print(f"Stored object {i}: {result['compressed_size']/1024:.1f}KB in {result['storage_location']}")
            
            # Print memory usage every 3 objects
            if (i + 1) % 3 == 0:
                stats = manager.get_storage_stats()
                memory_usage = stats['storage']['memory']['usage']
                memory_limit = stats['storage']['memory']['limit']
                memory_pct = stats['storage']['memory']['usage_pct']
                print(f"Memory usage: {memory_usage/1024/1024:.2f}MB / {memory_limit/1024/1024:.2f}MB ({memory_pct:.1f}%)")
        
        # Print storage distribution
        print("\nStorage distribution after initial storage:")
        stats = manager.get_storage_stats()
        for location, info in stats['storage'].items():
            if isinstance(info, dict) and 'usage' in info:
                print(f"  {location}: {info['usage']/1024/1024:.2f}MB ({info.get('items', 0)} items)")
        
        # Force optimization
        print("\nRunning optimization...")
        optimization_result = manager.optimize()
        print(f"Optimization result: {optimization_result}")
        
        # Print storage distribution after optimization
        print("\nStorage distribution after optimization:")
        stats = manager.get_storage_stats()
        for location, info in stats['storage'].items():
            if isinstance(info, dict) and 'usage' in info:
                print(f"  {location}: {info['usage']/1024/1024:.2f}MB ({info.get('items', 0)} items)")
        
        # Verify we can still retrieve data
        print("\nVerifying data retrieval...")
        for i in range(0, 15, 3):  # Check every 3rd object
            key = f"key_{i}"
            data = await manager.retrieve_data(key)
            if data:
                print(f"Successfully retrieved {key}, array shape: {data['data'].shape}")
            else:
                print(f"Failed to retrieve {key}")
        
        # Clean up
        manager.shutdown()
        
    except Exception as e:
        print(f"Error in test: {str(e)}")
        traceback.print_exc()
        manager.shutdown()