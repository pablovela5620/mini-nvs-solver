import PIL
import PIL.Image
import torch
from queue import SimpleQueue
from typing import Literal
from jaxtyping import Float64
from mini_nvs_solver.custom_diffusers_pipeline.svd import StableVideoDiffusionPipeline


def svd_render_threaded(
    image_o: PIL.Image.Image,
    masks: Float64[torch.Tensor, "b 72 128"],
    cond_image: PIL.Image.Image,
    lambda_ts: Float64[torch.Tensor, "n b"],
    num_denoise_iters: Literal[2, 25, 50, 100],
    weight_clamp: float,
    svd_pipe: StableVideoDiffusionPipeline,
    log_queue: SimpleQueue | None = None,
) -> None:
    frames: list[PIL.Image.Image] = svd_pipe(
        [image_o],
        log_queue=log_queue,
        temp_cond=cond_image,
        mask=masks,
        lambda_ts=lambda_ts,
        weight_clamp=weight_clamp,
        num_frames=25,
        decode_chunk_size=8,
        num_inference_steps=num_denoise_iters,
    ).frames[0]
    if log_queue is not None:
        log_queue.put(frames)


# to allow logging from a separate thread
# log_queue: SimpleQueue = SimpleQueue()
# handle = threading.Thread(
#     target=svd_render_threaded,
#     kwargs={
#         "image_o": rgb_resized,
#         "masks": masks,
#         "cond_image": cond_image,
#         "lambda_ts": lambda_ts,
#         "num_denoise_iters": num_denoise_iters,
#         "weight_clamp": 0.2,
#         "log_queue": None,
#     },
# )

# handle.start()
# i = 0
# while True:
#     msg = log_queue.get()
#     match msg:
#         case frames if all(isinstance(frame, PIL.Image.Image) for frame in frames):
#             break
#         case entity_path, entity, times:
#             i += 1
#             rr.reset_time()
#             for timeline, time in times:
#                 if isinstance(time, int):
#                     rr.set_time_sequence(timeline, time)
#                 else:
#                     rr.set_time_seconds(timeline, time)
#             static = False
#             if entity_path == "diffusion_step":
#                 static = True
#             rr.log(entity_path, entity, static=static)
#             yield stream.read(), None, [], f"{i} out of {num_denoise_iters}"
#         case _:
#             assert False
# handle.join()
