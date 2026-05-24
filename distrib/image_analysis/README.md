# Image Analysis Examples

These examples illustrate the use of low-rank image compression via the randomized SVD, QR factorization, and interpolative decomposition.
The examples also highlight the impact of oversampling and using a few power iterations.  
They also illustrate that while the error in the approximation can be of the same order, the different factorization techniques can provide very different quality of approximations to the eye.

## Scripts

### test_image_id (Method Comparison)

Compares five low-rank factorization methods on RGB images (reshaped to m x 3n matrix):

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

Python and MATLAB require initializing the computing environment.

In Python, run:

```bash
setup-venv.sh
```


In MATLAB, run:

```bash
startup.m
```

### Example Python run sequence 

In a terminal, type the following commands:

```bash
./setup-venv.sh 
python3 download_images.py 
python3 test_image_id.py
```

### Example MATLAB run sequence

In the MATLAB command line, type the following:

```bash
startup
download_images
test_image_id
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
