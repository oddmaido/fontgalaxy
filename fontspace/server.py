#!/usr/bin/env python3
"""
Backend server for Font Space Explorer.

Serves font data and generates blended font previews based on position.
"""

import io
import base64
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from scipy.spatial import cKDTree

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.autoencoder import GlyphAutoencoder


app = FastAPI(title="Font Space Explorer")

# Global state
model = None
embeddings = None
umap_coords = None  # 3D coordinates
font_ids = None
font_families = None
kdtree = None
device = None
glyph_grids_dir = None
cluster_labels = None


def load_data(
    checkpoint_path: Path,
    coords_path: Path,
    grids_dir: Path,
    latent_dim: int = 256,
):
    """Load model and data."""
    global model, embeddings, umap_coords, font_ids, font_families, kdtree, device, glyph_grids_dir

    glyph_grids_dir = grids_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = GlyphAutoencoder(latent_dim=latent_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Model loaded")

    # Load 3D coordinates
    print(f"Loading 3D coordinates from {coords_path}")
    coords_data = np.load(coords_path, allow_pickle=True)
    umap_coords = coords_data["coords_3d"]
    font_ids = list(coords_data["font_ids"])
    font_families = list(coords_data.get("font_families", ["Unknown"] * len(font_ids)))
    print(f"Loaded {len(umap_coords)} fonts with 3D coordinates")

    # Build KD-tree for fast nearest neighbor lookup
    kdtree = cKDTree(umap_coords)
    print("KD-tree built")


@app.get("/")
async def index():
    """Serve the main page."""
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/api/bounds")
async def get_bounds():
    """Get the coordinate bounds of the font space (3D)."""
    return {
        "min_x": float(umap_coords[:, 0].min()),
        "max_x": float(umap_coords[:, 0].max()),
        "min_y": float(umap_coords[:, 1].min()),
        "max_y": float(umap_coords[:, 1].max()),
        "min_z": float(umap_coords[:, 2].min()),
        "max_z": float(umap_coords[:, 2].max()),
    }


@app.get("/api/stars")
async def get_stars(
    min_x: float = Query(...),
    max_x: float = Query(...),
    min_y: float = Query(...),
    max_y: float = Query(...),
    limit: int = Query(1000),
):
    """Get stars (fonts) within a bounding box."""
    # Filter fonts within bounds
    mask = (
        (umap_coords[:, 0] >= min_x) & (umap_coords[:, 0] <= max_x) &
        (umap_coords[:, 1] >= min_y) & (umap_coords[:, 1] <= max_y)
    )
    indices = np.where(mask)[0]

    # Limit number of stars
    if len(indices) > limit:
        indices = np.random.choice(indices, limit, replace=False)

    stars = []
    for idx in indices:
        stars.append({
            "id": int(idx),
            "x": float(umap_coords[idx, 0]),
            "y": float(umap_coords[idx, 1]),
            "name": font_families[idx],
        })

    return {"stars": stars}


@app.get("/api/all_stars")
async def get_all_stars():
    """Get all stars for initial load (3D)."""
    stars = []
    for idx in range(len(umap_coords)):
        stars.append({
            "id": int(idx),
            "x": float(umap_coords[idx, 0]),
            "y": float(umap_coords[idx, 1]),
            "z": float(umap_coords[idx, 2]),
            "name": font_families[idx],
        })
    return {"stars": stars}


@torch.no_grad()
def blend_and_render(x: float, y: float, k: int = 5, chars: str = "AaBbCc123"):
    """Blend nearby fonts and render preview."""
    # Find k nearest neighbors
    distances, indices = kdtree.query([x, y], k=k)

    # Handle case where we're exactly on a point
    distances = np.maximum(distances, 1e-6)

    # Inverse distance weighting
    weights = 1.0 / distances
    weights = weights / weights.sum()

    # Blend embeddings
    blended_z = np.zeros(embeddings.shape[1])
    for idx, weight in zip(indices, weights):
        blended_z += weight * embeddings[idx]

    # Convert to tensor
    z = torch.from_numpy(blended_z).float().to(device)

    # Render each character
    char_images = []
    for char in chars:
        # Decode - we need a glyph embedding, but our autoencoder doesn't use glyph IDs
        # So we just decode the same z multiple times (the output will be the same)
        # Actually, our autoencoder doesn't condition on glyph - it just reconstructs
        # So we'll need to think about this differently...

        # For now, let's just decode and show what comes out
        recon = model.decode(z.unsqueeze(0))
        char_images.append(recon.squeeze().cpu().numpy())

    # Create grid of characters
    img_size = 64
    grid_cols = len(chars)
    grid = np.ones((img_size, img_size * grid_cols), dtype=np.float32)

    for i, img in enumerate(char_images):
        grid[:, i*img_size:(i+1)*img_size] = img

    # Convert to PIL and base64
    grid_uint8 = (grid * 255).astype(np.uint8)
    pil_img = Image.fromarray(grid_uint8, mode='L')

    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    b64 = base64.b64encode(buffer.getvalue()).decode()

    return b64


def load_font_grid(font_idx: int, glyph_size: int = 64) -> np.ndarray:
    """Load font grid image from disk."""
    fid = font_ids[font_idx]
    png_path = glyph_grids_dir / f"{fid}.png"

    if not png_path.exists():
        return None

    img = Image.open(png_path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


@app.get("/api/blend")
async def get_blended_font(
    x: float = Query(...),
    y: float = Query(...),
    z: float = Query(0.0),
    k: int = Query(5),
):
    """Get blended font preview at position (3D)."""
    # Find more neighbors than needed to allow deduplication
    distances, indices = kdtree.query([x, y, z], k=k * 3)
    distances = np.maximum(distances, 1e-6)

    # Deduplicate by font family name, keep k unique
    seen_names = set()
    unique_indices = []
    unique_distances = []
    for idx, dist in zip(indices, distances):
        name = font_families[idx]
        if name not in seen_names:
            seen_names.add(name)
            unique_indices.append(idx)
            unique_distances.append(dist)
            if len(unique_indices) >= k:
                break

    # Recalculate weights for unique neighbors using Gaussian falloff (smoother)
    unique_distances = np.array(unique_distances)
    sigma = np.median(unique_distances) + 0.1  # Adaptive sigma based on local density
    weights = np.exp(-unique_distances**2 / (2 * sigma**2))
    weights = weights / weights.sum()

    # Get neighbor info
    neighbors = []
    for idx, dist, weight in zip(unique_indices, unique_distances, weights):
        neighbors.append({
            "id": int(idx),
            "name": font_families[idx],
            "distance": float(dist),
            "weight": float(weight),
            "x": float(umap_coords[idx, 0]),
            "y": float(umap_coords[idx, 1]),
            "z": float(umap_coords[idx, 2]),
        })

    indices = unique_indices
    weights = weights

    # Load and blend actual font images
    blended_img = None
    for idx, weight in zip(indices, weights):
        grid = load_font_grid(idx)
        if grid is None:
            continue

        if blended_img is None:
            blended_img = grid * weight
        else:
            # Resize if shapes don't match
            if grid.shape != blended_img.shape:
                h, w = blended_img.shape
                grid_resized = np.array(Image.fromarray((grid * 255).astype(np.uint8)).resize((w, h))) / 255.0
                blended_img += grid_resized * weight
            else:
                blended_img += grid * weight

    if blended_img is None:
        return {"error": "No fonts found", "neighbors": neighbors}

    # Convert to base64 PNG
    img_uint8 = (blended_img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='L')

    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    b64 = base64.b64encode(buffer.getvalue()).decode()

    return {
        "image": b64,
        "neighbors": neighbors,
    }


@app.get("/api/font_grid/{font_id}")
async def get_font_grid(font_id: int):
    """Get font specimen grid image."""
    if font_id < 0 or font_id >= len(font_ids):
        return JSONResponse({"error": "Invalid font ID"}, status_code=404)

    grid = load_font_grid(font_id)
    if grid is None:
        return JSONResponse({"error": "Font grid not found"}, status_code=404)

    # Convert to base64 PNG
    img_uint8 = (grid * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='L')

    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    b64 = base64.b64encode(buffer.getvalue()).decode()

    return {
        "id": font_id,
        "name": font_families[font_id],
        "image": b64,
    }


@app.get("/api/font/{font_id}")
async def get_font(font_id: int):
    """Get a specific font's embedding decoded."""
    if font_id < 0 or font_id >= len(embeddings):
        return JSONResponse({"error": "Invalid font ID"}, status_code=404)

    z = torch.from_numpy(embeddings[font_id]).float().to(device).unsqueeze(0)
    with torch.no_grad():
        recon = model.decode(z)

    img = recon.squeeze().cpu().numpy()
    img_uint8 = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='L')

    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    b64 = base64.b64encode(buffer.getvalue()).decode()

    return {
        "id": font_id,
        "name": font_families[font_id],
        "font_id": font_ids[font_id],
        "x": float(umap_coords[font_id, 0]),
        "y": float(umap_coords[font_id, 1]),
        "image": b64,
    }


# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints_ae/best_model.pt"))
    parser.add_argument("--coords", type=Path, default=Path("font_coords_3d.npz"))
    parser.add_argument("--grids", type=Path, default=Path("glyph_grids"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Load data before starting server
    load_data(args.checkpoint, args.coords, args.grids)

    uvicorn.run(app, host=args.host, port=args.port)
