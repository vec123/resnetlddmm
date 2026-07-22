IRREPS_PRESETS = {
    "tiny": {
        "input": "1x0e",
        "intermediate": "8x0e + 8x1o",
        "output": "4x0e + 2x1o",
    },
    "small": {
        "input": "1x0e",
        "intermediate": "16x0e + 16x1o + 8x2e",
        "output": "8x0e + 2x1o",
    },
    "standard": {
        "input": "1x0e",
        "intermediate": "32x0e + 32x0o + 16x1e + 16x1o",
        "output": "8x0e + 2x1o",
    },
    "large": {
        "input": "1x0e",
        "intermediate": "64x0e + 64x0o + 32x1e + 32x1o + 16x2e + 16x2o",
        "output": "16x0e + 4x1o",
    },
}