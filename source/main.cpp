#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define HAS_NEON 1
#else
#define HAS_NEON 0
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

extern "C" {
#include "gifenc.h"
}

static inline int clampi(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

// ──────────────────────── grayscale conversion ────────────────────────

#if HAS_NEON
// Process 8 RGB pixels at a time using NEON fixed-point luma.
// Coefficients scaled to Q8: R*77 + G*150 + B*29 ≈ Rec.601 luma * 256
static void grayscale_rgb_neon(const uint8_t* __restrict src,
                               uint8_t* __restrict dst, size_t npixels) {
    const uint8x8_t cR = vdup_n_u8(77);
    const uint8x8_t cG = vdup_n_u8(150);
    const uint8x8_t cB = vdup_n_u8(29);

    size_t i = 0;
    for (; i + 8 <= npixels; i += 8) {
        // Deinterleave 8 RGB pixels → r[8], g[8], b[8]
        uint8x8x3_t rgb = vld3_u8(src + i * 3);
        uint16x8_t acc = vmull_u8(rgb.val[0], cR);
        acc = vmlal_u8(acc, rgb.val[1], cG);
        acc = vmlal_u8(acc, rgb.val[2], cB);
        // Shift right by 8 (divide by 256) and narrow to u8
        uint8x8_t gray = vshrn_n_u16(acc, 8);
        vst1_u8(dst + i, gray);
    }
    // Scalar tail
    for (; i < npixels; ++i) {
        const uint8_t* p = src + i * 3;
        dst[i] = (uint8_t)((77 * (int)p[0] + 150 * (int)p[1] + 29 * (int)p[2]) >> 8);
    }
}

static void grayscale_rgba_neon(const uint8_t* __restrict src,
                                uint8_t* __restrict dst, size_t npixels) {
    const uint8x8_t cR   = vdup_n_u8(77);
    const uint8x8_t cG   = vdup_n_u8(150);
    const uint8x8_t cB   = vdup_n_u8(29);
    const uint16x8_t c255 = vdupq_n_u16(255);

    size_t i = 0;
    for (; i + 8 <= npixels; i += 8) {
        uint8x8x4_t rgba = vld4_u8(src + i * 4);
        uint8x8_t a8 = rgba.val[3];
        uint16x8_t a  = vmovl_u8(a8);
        uint16x8_t ia = vsubq_u16(c255, a); // 255 - alpha

        // Composite each channel over white: out = (c*a + 255*(255-a)) / 255
        // Since max value = 255*255 = 65025, it fits u16.
        uint16x8_t rr = vaddq_u16(vmulq_u16(vmovl_u8(rgba.val[0]), a), vmulq_u16(c255, ia));
        uint16x8_t gg = vaddq_u16(vmulq_u16(vmovl_u8(rgba.val[1]), a), vmulq_u16(c255, ia));
        uint16x8_t bb = vaddq_u16(vmulq_u16(vmovl_u8(rgba.val[2]), a), vmulq_u16(c255, ia));

        // Divide by 255 ≈ (v + 128) >> 8
        uint16x8_t half = vdupq_n_u16(128);
        uint8x8_t r8 = vshrn_n_u16(vaddq_u16(rr, half), 8);
        uint8x8_t g8 = vshrn_n_u16(vaddq_u16(gg, half), 8);
        uint8x8_t b8 = vshrn_n_u16(vaddq_u16(bb, half), 8);

        // Luma
        uint16x8_t lum = vmull_u8(r8, cR);
        lum = vmlal_u8(lum, g8, cG);
        lum = vmlal_u8(lum, b8, cB);
        uint8x8_t gray = vshrn_n_u16(lum, 8);
        vst1_u8(dst + i, gray);
    }
    // Scalar tail
    for (; i < npixels; ++i) {
        const uint8_t* p = src + i * 4;
        int r = p[0], g = p[1], b = p[2], a = p[3];
        int ia = 255 - a;
        r = (r * a + 255 * ia + 127) / 255;
        g = (g * a + 255 * ia + 127) / 255;
        b = (b * a + 255 * ia + 127) / 255;
        dst[i] = (uint8_t)clampi((77 * r + 150 * g + 29 * b) >> 8, 0, 255);
    }
}
#endif // HAS_NEON

static std::vector<uint8_t> to_grayscale(const uint8_t* pixels, int w, int h, int comp) {
    const size_t npixels = (size_t)w * (size_t)h;
    std::vector<uint8_t> gray(npixels);

#if HAS_NEON
    if (comp == 3) {
        grayscale_rgb_neon(pixels, gray.data(), npixels);
        return gray;
    }
    if (comp == 4) {
        grayscale_rgba_neon(pixels, gray.data(), npixels);
        return gray;
    }
#endif

    // Scalar fallback
    for (size_t idx = 0; idx < npixels; ++idx) {
        const uint8_t* p = pixels + idx * comp;
        int r = p[0], g = p[1], b = p[2];
        if (comp == 4) {
            int a = p[3], ia = 255 - a;
            r = (r * a + 255 * ia + 127) / 255;
            g = (g * a + 255 * ia + 127) / 255;
            b = (b * a + 255 * ia + 127) / 255;
        }
        int lum = (299 * r + 587 * g + 114 * b + 500) / 1000;
        gray[idx] = (uint8_t)clampi(lum, 0, 255);
    }
    return gray;
}

// ──────────────────────── Atkinson dithering ────────────────────────
// Optimised: uses int16_t error buffer (halves cache footprint vs int),
// eliminates per-pixel bounds checks with a 2-pixel border padding,
// and avoids lambda/function-call overhead.

static std::vector<uint8_t> atkinson_1bit(const std::vector<uint8_t>& gray, int w, int h, int threshold) {
    // Padded buffer: 2 extra columns on each side, 2 extra rows on bottom.
    // This lets us skip all bounds checks in the inner loop.
    const int pw = w + 4;                       // padded width
    const int ph = h + 2;                       // padded height
    const size_t psize = (size_t)pw * (size_t)ph;
    std::vector<int16_t> buf(psize, 0);

    // Copy gray values into padded buffer (offset by 2 columns)
    for (int y = 0; y < h; ++y) {
        const uint8_t* src = gray.data() + (size_t)y * (size_t)w;
        int16_t*       dst = buf.data()  + (size_t)y * (size_t)pw + 2;
        for (int x = 0; x < w; ++x)
            dst[x] = (int16_t)src[x];
    }

    std::vector<uint8_t> out((size_t)w * (size_t)h);

    for (int y = 0; y < h; ++y) {
        int16_t* __restrict row0 = buf.data() + (size_t)y       * (size_t)pw + 2; // current row, offset to real pixel 0
        int16_t* __restrict row1 = row0 + pw;                                      // y+1
        int16_t* __restrict row2 = row1 + pw;                                      // y+2
        uint8_t* __restrict orow = out.data() + (size_t)y * (size_t)w;

        for (int x = 0; x < w; ++x) {
            int oldp = row0[x];
            // Clamp (error diffusion can push values outside [0,255])
            oldp = (oldp < 0) ? 0 : (oldp > 255) ? 255 : oldp;

            int newp = (oldp >= threshold) ? 255 : 0;
            int e    = (oldp - newp) >> 3;   // err / 8  (arithmetic shift)

            orow[x] = (newp == 255) ? 1 : 0;

            // Diffuse error — no bounds checks needed thanks to padding
            row0[x + 1] += (int16_t)e;
            row0[x + 2] += (int16_t)e;
            row1[x - 1] += (int16_t)e;
            row1[x    ] += (int16_t)e;
            row1[x + 1] += (int16_t)e;
            row2[x    ] += (int16_t)e;
        }
    }

    return out;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage: %s <input.(png|jpg|jpeg)> <output.gif> [threshold 0-255]\n"
            "Example: %s in.png out.gif 128\n",
            argv[0], argv[0]);
        return 2;
    }

    const char* in_path  = argv[1];
    const char* out_path = argv[2];
    int threshold = 128;
    if (argc >= 4) threshold = clampi(std::atoi(argv[3]), 0, 255);

    int w = 0, h = 0, comp = 0;
    stbi_uc* pixels = stbi_load(in_path, &w, &h, &comp, 0);
    if (!pixels) {
        std::fprintf(stderr, "Failed to load image: %s\n", in_path);
        return 1;
    }
    if (!(comp == 3 || comp == 4)) {
        std::fprintf(stderr, "Unsupported channel count %d (need RGB or RGBA)\n", comp);
        stbi_image_free(pixels);
        return 1;
    }

    auto gray = to_grayscale(pixels, w, h, comp);
    stbi_image_free(pixels);

    auto indices = atkinson_1bit(gray, w, h, threshold);

    // 4-colour palette (depth=2): gifenc clamps depth to min 2 internally,
    // so we must supply 4 entries.  Index 0 = black, 1 = white, 2-3 = unused (black).
    uint8_t palette[12] = {
        0x00, 0x00, 0x00,   // 0: black
        0xFF, 0xFF, 0xFF,   // 1: white
        0x00, 0x00, 0x00,   // 2: (unused)
        0x00, 0x00, 0x00,   // 3: (unused)
    };

    // depth=2 => 4 palette slots; bgindex=-1 => no transparent colour; loop=-1 => no loop ext
    ge_GIF* gif = ge_new_gif(out_path, (uint16_t)w, (uint16_t)h, palette, 2, -1, -1);
    if (!gif) {
        std::fprintf(stderr, "Failed to create GIF: %s\n", out_path);
        return 1;
    }

    std::memcpy(gif->frame, indices.data(), (size_t)w * (size_t)h);
    ge_add_frame(gif, 0);   // single frame; delay ignored
    ge_close_gif(gif);

    return 0;
}
