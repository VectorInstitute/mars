# Quickstart

Install MARScope from the repository and analyze your first video. The models (about 0.8 GB each)
download from Hugging Face the first time they are used.

=== "Web UI (GPU)"

    ```bash
    git clone https://github.com/VectorInstitute/marscope.git
    cd marscope
    uv sync --extra cuda
    uv run --extra cuda marscope --serve
    # open http://localhost:8000/app
    ```

=== "Web UI (Mac or CPU)"

    ```bash
    git clone https://github.com/VectorInstitute/marscope.git
    cd marscope
    uv sync
    uv run marscope --serve
    # open http://localhost:8000/app
    ```

=== "Command line"

    ```bash
    uv run --extra cuda marscope --video clip.mp4                        # MARS
    uv run --extra cuda marscope --video clip.mp4 --models all --report pdf
    ```

=== "Python"

    ```python
    from marscope import MARScope
    from marscope.analysis.ranges import rarity

    scope = MARScope(models="mars,gend")
    result = scope.analyze("clip.mp4")
    for model in result.models.values():
        print(model.name, f"probability of fake {model.fake:.0%}", rarity(model.context.real_rank, model.name))
    ```

In the app: choose a video (up to 2 minutes and 1080p), and press **Analyze**. MARS runs by default; tick
GenD and DFD-HR under **Choose models** to compare. By
default you are asked to guess real or fake before any score is shown.

On a remote GPU machine or a cluster, see [Web UI and serving](guides/serve.md). For platforms and
GPU setup, see [Installation](guides/install.md).
