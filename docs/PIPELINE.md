- All Steps (except warping) should be applied using cv2 for simplicity -

Example for IAM:

```json
[
  {
    "load": "greyscale",
    "invert": true,
    "threshold": true,
    "crop": "height",
    "scale": [-1, 124],
    "save": "temp"
  },
  "maxwidth",
  {
    "load": "greyscale",
    "fill": ["maxwidth", -1],
    "warp": {
      "num": 5,
      "gridsize": [30, 30],
      "deviation": 2.7
    },
    "save": "processed"
  }
]
```
