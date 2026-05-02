# taxispy
A Python-based Software for the Quantitative Analysis of Bacterial Chemotaxis

## Docker

Build the notebook image:

```bash
docker build -t taxispy .
```

Run Jupyter with the repository mounted at `/Documents`:

```bash
docker run --rm -p 8888:8888 -v "$PWD:/Documents" taxispy
```

The image is based on the multi-architecture Jupyter Docker Stack and is intended
to build for both `linux/amd64` and `linux/arm64`. When run without a bind mount,
the image opens with the original TaxisPy tutorial workspace in `/Documents`.
