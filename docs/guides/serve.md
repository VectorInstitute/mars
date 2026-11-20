# Web UI and serving

```bash
uv run --extra cuda marscope --serve
```

Open **http://localhost:8000/app** (the landing page is at `/`). The same server exposes the
[REST API](rest.md).

| Option | Default | What it does |
|---|---|---|
| `--host` | `127.0.0.1` | Interface to listen on. The default keeps the app reachable from this machine only |
| `--port` | `8000` | Port |
| `--models` | `mars` | Models loaded at start; the others load when picked on the start screen |
| `--precision` | `auto` | `fp16` on the GPU unless `fp32` is asked for; the CPU always runs fp32 |
| `--device` | `auto` | `cuda` or `cpu` to force one |

## Using the app

1. **Choose a video** (MP4, MOV or WebM, up to 2 minutes and 1080p), pick the models, and press
   **Analyze**. With **Guess first** on, you guess real or fake before any score is shown.
2. **While it runs**, one card per model fills in clip by clip, then each test lights up as it runs.
3. **The result:** a frame viewer with the model's map, a timeline with every clip's score per
   model, and four tabs:
   - **Check:** the evidence and the tests against chance. The common test is computed in the
     background while you look around, so pressing it usually answers at once; nothing is shown
     until you press it.
   - **Notes:** how far to trust the models (A1–A5), provenance, and laws that may apply, by region.
   - **What if:** JPEG quality, resolution, blur, noise, or "shared many times", and every score
     again.
   - **Learn:** the five assumptions, the MIT AI Risk Repository, how far one score goes, the model
     cards, and Ask.
4. **Report:** the report sheet, with Copy, Download (PDF, HTML, Markdown, session JSON) and the
   evidence card. See [Reports and Ask](report.md).

The app is in English and French, with light and dark themes.

## On a remote GPU machine or a cluster

Run the server on the GPU machine and let it listen on its network interface:

```bash
uv run --extra cuda marscope --serve --host 0.0.0.0
```

Then, on your laptop, forward a local port to it, through the login node if there is one:

```bash
ssh -N -L 8010:<gpu-node>:8000 <user>@<login-host>
```

and open **http://localhost:8010/app**. After you log in, the command stays silent: the tunnel is
open until you stop it.

- `<gpu-node>` is the machine the server runs on (on Slurm, the node name `squeue` shows).
- VS Code and Cursor can forward the port for you from the **Ports** tab instead.

!!! tip "\"Address already in use\""
    Something on your laptop already holds that local port: often an older tunnel still running in
    another terminal, or your editor's own port forwarding. Pick another local port (`-L 8020:...`),
    or find what holds it with `lsof -nP -iTCP -sTCP:LISTEN` and close it.

!!! warning "Shared machines"
    With `--host 0.0.0.0`, other users of the network can reach the app. Jobs and their videos are
    deleted an hour after last use, or when you leave the result.

## Limits

Uploads are checked before a job is created: the file type by its content, at most 500 MB, 2
minutes and 1080p. The server runs one analysis at a time per GPU and queues the rest.
