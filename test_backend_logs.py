import asyncio
import os
from orchestrator import EmotionOrchestrator

async def test_backend_logs():
    print("--- Starting Backend Log Test ---")
    orchestrator = EmotionOrchestrator()
    
    # Try to find a file to analyze
    file_path = orchestrator.get_next_file()
    if not file_path:
        print("No files found in 'aufnahmen'. Please ensure there is at least one .mp4 file.")
        return

    print(f"Testing with file: {file_path}")
    
    try:
        # We mock the status callback to see if it's being called
        def status_cb(msg):
            print(f"[STATUS CALLBACK]: {msg}")
            
        result = await orchestrator.run_analysis(file_path=file_path, status_callback=status_cb)
        print("--- Analysis Result ---")
        print(result)
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_backend_logs())
