import torch
from pathlib import Path
from qwenimg2512.config import Config
from qwenimg2512.worker import GenerationWorker

if __name__ == "__main__":
    config = Config.load()
    
    # Modify settings in memory
    config.generation.prompt = "A cute cat, high quality, highly detailed"
    config.generation.num_inference_steps = 20
    config.generation.schedule_name = "bong_tangent"
    config.generation.sampler_name = "euler"
    config.generation.output_dir = "/tmp/qwen_test"
    
    Path("/tmp/qwen_test").mkdir(parents=True, exist_ok=True)
    
    worker = GenerationWorker(config.generation, config.model_paths)
    
    # We will just run it synchronously for testing
    print("Running worker with bong_tangent...")
    worker.run()
    print("Done!")
