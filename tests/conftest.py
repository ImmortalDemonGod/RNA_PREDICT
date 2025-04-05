"""
Pytest configuration and plugins.
"""

import pytest
import psutil
import os
import time
import faulthandler
from datetime import datetime

# Enable faulthandler with a timeout
faulthandler.enable()
# Set timeout to 60 seconds
faulthandler.dump_traceback_later(60, repeat=False)

def get_memory_usage():
    """Get current memory usage of the Python process."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert to MB

@pytest.fixture(autouse=True)
def memory_tracker(request):
    """Track memory usage for each test automatically."""
    start_mem = get_memory_usage()
    start_time = time.time()
    
    yield
    
    end_mem = get_memory_usage()
    end_time = time.time()
    
    mem_diff = end_mem - start_mem
    duration = end_time - start_time
    
    print(f"\nMemory usage for {request.node.name}:")
    print(f"  Before: {start_mem:.2f} MB")
    print(f"  After:  {end_mem:.2f} MB")
    print(f"  Change: {mem_diff:+.2f} MB")
    print(f"  Duration: {duration:.2f}s")

class MemoryUsagePlugin:
    """Plugin to track and display memory usage during test execution."""
    
    def __init__(self):
        self.memory_usage = {}
        # Create memory_logs directory if it doesn't exist
        self.log_dir = "memory_logs"
        os.makedirs(self.log_dir, exist_ok=True)
    
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        """Track memory usage before and after each test."""
        if call.when == "call":
            # Record memory usage before test
            start_mem = get_memory_usage()
            start_time = time.time()
            
            outcome = yield
            
            # Record memory usage after test
            end_mem = get_memory_usage()
            end_time = time.time()
            
            mem_diff = end_mem - start_mem
            duration = end_time - start_time
            
            print(f"\nMemory usage for {item.name}:")
            print(f"  Before: {start_mem:.2f} MB")
            print(f"  After:  {end_mem:.2f} MB")
            print(f"  Change: {mem_diff:+.2f} MB")
            print(f"  Duration: {duration:.2f}s")
            
            # Store the results for the summary
            self.memory_usage[item.nodeid] = {
                'before': start_mem,
                'after': end_mem,
                'duration': duration
            }
        else:
            outcome = yield
    
    def pytest_terminal_summary(self, terminalreporter, exitstatus, config):
        """Display memory usage summary at the end of the test run."""
        if not self.memory_usage:
            return
            
        terminalreporter.write("\nMemory Usage Summary:\n")
        terminalreporter.write("=" * 80 + "\n")
        
        # Sort tests by memory usage
        sorted_tests = sorted(
            self.memory_usage.items(),
            key=lambda x: x[1]['after'] - x[1]['before'],
            reverse=True
        )
        
        for test_id, mem_info in sorted_tests:
            mem_diff = mem_info['after'] - mem_info['before']
            terminalreporter.write(
                f"{test_id}\n"
                f"  Memory: {mem_diff:+.2f} MB (before: {mem_info['before']:.2f} MB, "
                f"after: {mem_info['after']:.2f} MB)\n"
                f"  Duration: {mem_info['duration']:.2f}s\n"
            )
        
        # Save summary to file in memory_logs directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(self.log_dir, f"memory_usage_{timestamp}.txt")
        with open(summary_file, "w") as f:
            f.write("Memory Usage Summary:\n")
            f.write("=" * 80 + "\n")
            for test_id, mem_info in sorted_tests:
                mem_diff = mem_info['after'] - mem_info['before']
                f.write(
                    f"{test_id}\n"
                    f"  Memory: {mem_diff:+.2f} MB (before: {mem_info['before']:.2f} MB, "
                    f"after: {mem_info['after']:.2f} MB)\n"
                    f"  Duration: {mem_info['duration']:.2f}s\n"
                )

def pytest_configure(config):
    """Register the memory usage plugin."""
    config.pluginmanager.register(MemoryUsagePlugin()) 