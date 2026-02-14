import torch
import numpy as np
import tyro
from rich import print
import os
import inspect
from einops import rearrange
from cotracker.utils.visualizer import Visualizer
import imageio.v3 as iio

def main(
    rgb_path: str = "demos/box_video_2/rgb.mp4",
    mask_path: str = "demos/box_video_2/sam3_outputs/mask.npy",
    output_dir: str = "demos/box_video_2/cotracker3_outputs",
    grid_size: int = 80,
):
    print(f"[bold green]===============================[/bold green]")
    print(f"[bold green]Output directory: {output_dir}[/bold green]")
    print(f"[bold green]===============================[/bold green]")

    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the video
    rgb = iio.imread(rgb_path)
    
    first_frame_mask = np.load(mask_path)
    first_frame_mask = torch.from_numpy(first_frame_mask).float().to(device)
    first_frame_mask = rearrange(first_frame_mask, 'h w -> 1 1 h w')

    video = torch.from_numpy(rgb).float().to(device)  # B T C H W
    video = rearrange(video, 't h w c -> 1 t c h w')

    # Run Offline CoTracker:
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size, segm_mask=first_frame_mask) # B T N 2,  B T N 1

    print(f"[bold cyan]Visualizing results...[/bold cyan]")
    vis = Visualizer(save_dir=output_dir, linewidth=1, tracks_leave_trace=-1)
    vis.visualize(video, pred_tracks, pred_visibility)

    pred_tracks = pred_tracks.squeeze(0)
    pred_visibility = pred_visibility.squeeze(0)

    print(f"[bold cyan]Predicted Tracks: {pred_tracks.shape}[/bold cyan]")
    print(f"[bold cyan]Predicted Visibility: {pred_visibility.shape}[/bold cyan]")

    # save the predicted tracks and visibility
    np.savez_compressed(os.path.join(output_dir, "result.npz"), track=pred_tracks.cpu().numpy(), visibility=pred_visibility.cpu().numpy())

    print(f"[bold green]===============================[/bold green]")
    print(f"[bold green]CoTracker3 Done[/bold green]")
    print(f"[bold green]===============================[/bold green]")

if __name__ == "__main__":
    tyro.cli(main)