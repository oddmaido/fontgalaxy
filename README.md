# Font Galaxy

An interactive typographic space where fonts are translated into a navigable 3D galaxy. Each typeface exists as a point in space, defined by its visual characteristics encoded by a neural network autoencoder and projected using UMAP.

By moving through this environment, users explore typography not as static text, but as a system of relationships, distances, and transformations. Fonts can be selected, compared, and morphed into one another, revealing typography as a fluid, computational medium.

## Features

- **3D Font Space**: Navigate through thousands of fonts positioned by visual similarity
- **Font Blending**: See real-time interpolations between nearby fonts based on your position
- **Font Specimens**: Click on any font to view its full glyph grid
- **Smooth Navigation**: WASD/arrow keys to move, mouse to look, scroll to adjust speed

## Architecture

- **Autoencoder**: Encodes 64x64 glyph images into 256-dimensional latent vectors
- **UMAP**: Projects font embeddings into 3D coordinates for visualization
- **Backend**: FastAPI server with KD-tree for fast nearest-neighbor lookups
- **Frontend**: Three.js for WebGL rendering

## Requirements

- Python 3.10+
- PyTorch
- FastAPI, Uvicorn
- NumPy, SciPy, Pillow

## Data Files (not included)

The following files are required but not included in this repository due to size:

- `checkpoints_ae/best_model.pt` - Trained autoencoder weights
- `font_coords_3d.npz` - UMAP 3D coordinates for all fonts
- `glyph_grids/` - Directory of font specimen images

## Usage

```bash
# Install dependencies
pip install torch fastapi uvicorn numpy scipy pillow

# Run the server
python fontspace/server.py --checkpoint checkpoints_ae/best_model.pt \
                           --coords font_coords_3d.npz \
                           --grids glyph_grids

# Open http://localhost:8000 in your browser
```

## Controls

| Key | Action |
|-----|--------|
| W/A/S/D | Move through space |
| Mouse | Look around |
| Scroll | Adjust speed |
| Click | Select font at crosshair |
| ESC/Q | Close specimen modal |

## Project

Project by Mai Do, 2025/26
Developed at Bauhaus Typography
