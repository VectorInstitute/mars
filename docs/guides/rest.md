# REST API

`marscope --serve` exposes the API the web UI uses. Interactive docs are at
**http://localhost:8000/api/docs**.

## Analyze a video

```bash
# upload: returns {"id": "...", "status": "queued"}
curl -s -F video=@clip.mp4 -F models=mars,gend http://localhost:8000/jobs

# progress as server-sent events, until "done"
curl -N http://localhost:8000/jobs/<id>/events

# the result
curl -s http://localhost:8000/jobs/<id>

# the report as PDF
curl -s -X POST -H 'content-type: application/json' -d '{"format": "pdf"}' \
     http://localhost:8000/jobs/<id>/report -o report.pdf

# delete the video and the result now
curl -s -X DELETE http://localhost:8000/jobs/<id>
```

## Endpoints

| Endpoint | What it does |
|---|---|
| `GET /health` | Status, GPU, free memory, whether a language model is set up |
| `GET /models` · `GET /benchmark` · `GET /references` | Models (memory, typical scores), benchmark results, the reference list |
| `POST /jobs` | Upload a video (form fields `video`, `models`, `jurisdiction`, `language`); returns a job id |
| `GET /jobs/{id}/events` | Progress (server-sent events): stages, clips, tests |
| `GET /jobs/{id}` | The result: scores, evidence, notes |
| `GET /jobs/{id}/frames/{n}.jpg` · `GET /jobs/{id}/maps/{model}/{clip}` | Frame thumbnails; MARS patch maps |
| `POST /jobs/{id}/rescore` | Score one clip again with some frames removed (every model) |
| `POST /jobs/{id}/sweep` · `POST /jobs/{id}/probe` | A model's top frames or areas removed, against random (computed once per job; `record: false` keeps a background run out of the report) |
| `POST /jobs/{id}/occlusion` | An 8 × 8 occlusion map for one clip |
| `POST /jobs/{id}/edit` · `POST /jobs/{id}/quality-sweep` | What if edits on one clip; the same at JPEG 100 to 10 |
| `POST /jobs/{id}/notes` | The notes again for another region or language (scores do not change) |
| `POST /jobs/{id}/report` | The report as JSON, Markdown, HTML or PDF (`format`, `language`, optional `frame`) |
| `POST /jobs/{id}/ask` | A suggested question (`chip`) or your own (`question`) |
| `DELETE /jobs/{id}` | Delete the video and the result now |

Errors come back as `{"error": "..."}` with a 4xx status (for example 400 for a file that is not a
video, 404 for an unknown job, 409 while a job is still running).
