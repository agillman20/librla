# Image Analysis Examples

Low-rank image compression using randomized SVD and interpolative decomposition.

## Scripts

### test_image (Grayscale)

Compresses grayscale images using truncated SVD:

| Method | Description |
|--------|-------------|
| `svd_sketch(k)` | Basic rank-k approximation |
| `svd_sketch(k, extra, power)` | With oversampling and power iteration |

### test_image_rgb (RGB)

Same as above but for RGB images (reshaped to m x 3n matrix).

### test_image_id (Method Comparison)

Compares five low-rank approximation methods on RGB images:

| Method | Description |
|--------|-------------|
| `svd_sketch` | Randomized SVD |
| `id_sketch` | Interpolative decomposition |
| `svd_sketch` + extras | SVD with oversampling and power iteration |
| `id_sketch` + extras | ID with oversampling and power iteration |
| `qr_sketch` | QR with column pivoting |

ID selects k skeleton columns and expresses remaining columns as linear combinations.

## Prerequisites

Download sample images:
```bash
python download_images.py   # or .m / .jl
```

## Requirements

| Language | Packages |
|----------|----------|
| Python | numpy, scipy, matplotlib, pillow |
| MATLAB | Image Processing Toolbox (or Octave image package) |
| Julia | Images, FileIO, GLMakie |

## Image Sources

Image: pexels-anniroenkae-4793404.jpg
Image source: https://www.pexels.com/photo/a-colorful-painting-4793404/
License: https://www.pexels.com/license/

Image: pexels-flickr-149387.jpg
Image source: https://www.pexels.com/photo/silver-metal-round-gears-connected-to-each-other-149387/
License: Creative Commons Zero

Image: pexels-andre-ulysses-de-salis-2100065-7824822.jpg
Image source: https://www.pexels.com/photo/majestic-waterfalls-from-a-rocky-mountain-7824822/
License: https://www.pexels.com/license/
