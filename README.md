# C++ Atkinson dither

This is just hacked together, you should not use it there are much better options. Optimized to use Neon for SIMD on arm64 or SSE on x86

## Build

Requires a C/C++ toolchain (GCC or Clang) with C++17 support.

```sh
make
```

The binary is output to `dist/dither`.

To remove build artifacts:

```sh
make clean
```

## Usage

```sh
./dist/dither <input.(png|jpg|jpeg)> <output.gif> [threshold 0-255]
```

- **input** — any PNG, JPEG, or other image format supported by stb_image
- **output** — path for the resulting 1-bit dithered GIF
- **threshold** — optional binarization threshold (default: 128)

Example:

```sh
./dist/dither photo.jpg dithered.gif 128
```