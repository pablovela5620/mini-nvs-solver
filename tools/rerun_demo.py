import rerun as rr
from argparse import ArgumentParser
from pathlib import Path
from mini_nvs_solver.api.inference import nvs_solver_inference


if __name__ == "__main__":
    parser = ArgumentParser(description="NVS-Solver Rerun Demo")
    parser.add_argument(
        "--image-path",
        type=Path,
        default="/home/pablo/0Dev/docker/.per/repos/NVS_Solver/example_imgs/single/000001.jpg",
        help="Input Image FIle",
    )
    parser.add_argument(
        "--num-denoising-iterations",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        help="Output Path",
    )
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "nvs_solver")
    nvs_solver_inference(
        image_path=args.image_path,
        num_denoise_iters=args.num_denoising_iterations,
        direction="right",
        degrees_per_frame=0.5,
        major_radius=60.0,
        minor_radius=70.0,
        num_frames=25,
        save_path=args.save_path,
    )
    rr.script_teardown(args)
