import torch
from qwenimg2512.pipeline_patch import apply_custom_schedule
from qwenimg2512.schedules import get_bong_tangent_schedule, get_beta57_schedule
from diffusers import FlowMatchEulerDiscreteScheduler

class DummyConfig:
    num_train_timesteps = 1000

class DummyScheduler:
    def __init__(self):
        self.config = DummyConfig()
        self.sigmas = None
        self.timesteps = None

    def set_timesteps(self, num_inference_steps=50, device=None, **kwargs):
        pass

class DummyPipe:
    def __init__(self):
        self.scheduler = DummyScheduler()


if __name__ == "__main__":
    pipe = DummyPipe()
    num_steps = 10

    print("Testing bong_tangent context manager...")
    with apply_custom_schedule(pipe, "bong_tangent"):
        pipe.scheduler.set_timesteps(num_inference_steps=num_steps, device="cpu")
        print("Sigmas shape:", pipe.scheduler.sigmas.shape)
        print("Timesteps shape:", pipe.scheduler.timesteps.shape)
        assert len(pipe.scheduler.sigmas) == num_steps + 1
        assert pipe.scheduler.sigmas[-1].item() == 0.0

    print("\nTesting beta57 context manager...")
    with apply_custom_schedule(pipe, "beta57"):
        pipe.scheduler.set_timesteps(num_inference_steps=num_steps, device="cpu")
        print("Sigmas shape:", pipe.scheduler.sigmas.shape)
        print("Timesteps shape:", pipe.scheduler.timesteps.shape)
        assert len(pipe.scheduler.sigmas) == num_steps + 1
        assert pipe.scheduler.sigmas[-1].item() == 0.0

    print("\nEverything looks correct!")
