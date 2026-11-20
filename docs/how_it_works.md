# How it works

MARScope is a pipeline around three ONNX detectors. The detectors are unchanged research checkpoints;
what MARScope adds is everything around the score: the frames are prepared exactly as in training,
the score is placed among the model's test videos, the model is tested, and fixed rules explain the
result. This page walks through what happens between choosing a video and reading the report.

---

## The pipeline at a glance

<div class="ms-flow" markdown>

<div class="ms-flow__step" markdown>
**Video**{ .ms-flow__title }

MP4, MOV or WebM. Checked first: faces, bitrate, length and Content Credentials.
</div>

<div class="ms-flow__arrow">→</div>

<div class="ms-flow__step" markdown>
**Clips**{ .ms-flow__title }

Every second frame, in 64-frame clips, prepared exactly as in training.
</div>

<div class="ms-flow__arrow">→</div>

<div class="ms-flow__step" markdown>
**Models**{ .ms-flow__title }

MARS, GenD and DFD-HR score each clip (ONNX Runtime, GPU or CPU).
</div>

<div class="ms-flow__arrow">→</div>

<div class="ms-flow__step" markdown>
**Context and tests**{ .ms-flow__title }

Each score placed among the model's test videos; A1, A4 and A5 re-score the strongest clips.
</div>

<div class="ms-flow__arrow">→</div>

<div class="ms-flow__step ms-flow__step--out" markdown>
**Notes and report**{ .ms-flow__title }

Fixed, cited notes from deterministic rules; a report with optional LLM prose.
</div>

</div>

---

## Frames and clips

The models only give meaningful scores on input prepared the way they were trained, so this part
never changes:

- Frames are decoded as RGB with PyAV, and every second frame is kept.
- Kept frames are grouped in clips of 64. A short last clip is padded by repeating its last frame,
  and only its real frames count.
- Each frame is resized to a square (bicubic, no crop) and scaled to [-1, 1].
- A **clip score** is the model's probability of fake for that clip. The **video score** is the mean
  over clips.

Clip length, frame stride and image size are read from each model's ONNX metadata, not written in
the code. On a GPU the models run in fp16 (the same results as the paper); on the CPU, in fp32.

## Scores in context

A raw score means little: one model may give most real videos 10% and another 50%. So each model
comes with its **usual ranges**: the scores it gave the benchmark test videos (11,098 real, 12,435
fake), stored as 101 percentiles for each precision.

- "**Very unusual for a real video: fewer than 1 in 100 real videos score this high**" says where the
  score sits among the model's real test videos, in plain odds. The app draws it as a bar: the band is
  where real videos usually score (from the lowest 5% to the highest 5%), the dot is this video.
- A **high fake score** is one above the model's usual range for real videos (its 95th percentile).
  It describes the score; nothing is decided by it.
- Models are compared by that place, never by raw score. The result says whether the models agree,
  and how consistent the evidence is (*consistent*, *mixed* or *weak*).

## Evidence

- **MARS** has built-in evidence: how much each frame and each area (a 24 × 24 grid) adds to its
  score. The app draws it on the frame and on the timeline.
- **GenD and DFD-HR** have no built-in map. An **occlusion map** hides each area of an 8 × 8 grid in
  turn and measures how the score changes.
- **Tests against chance.** "Hide its top frames" (or areas) removes what the model ranks highest
  and compares with removing as many at random. If the top picks move the score much more than
  random ones, the model relies on them.

Evidence describes what a model relies on, never where a forgery is.

## The assumptions, A1 to A5

Every detector rests on assumptions about what gives a fake away. MARScope asks five of them as
questions, answered per model with measured numbers:

| | Question | How it is answered |
|---|---|---|
| **A1** | Does it break when the face outline (the seam of a face swap) is hidden? | The face outline is blanked in the strongest clips, which are scored again. Needs a face |
| **A2** | Does it break on a new generator? | The model's results on generators it never saw (GenVideo, DF26) |
| **A3** | Does it hold on videos unlike its training data? | The checks before the models: faces, bitrate, length, shape |
| **A4** | Does it break when the video is shared many times? | The strongest clips at 40% resolution, light blur and noise, JPEG quality 35 |
| **A5** | Does it break when fine pixel detail is removed? | The strongest clips with their fine detail removed |

The answer is judged against the model's own usual range for real videos:

- **Holds:** the score stays where it was.
- **Partly:** it falls, but stays higher than the model usually gives real videos.
- **Breaks:** it falls into the model's usual range for real videos, so the model needed what the
  test removed.

## Notes

Deterministic rules turn the measured values into notes, in English or French:

- **How far to trust the models:** A1–A5, per model.
- **Where the video comes from:** whether the file carries Content Credentials (C2PA), and what they
  say. Missing credentials prove nothing either way.
- **Laws and risks that may apply:** labelling rules in the EU, intimate-image laws in Canada and
  the US, and the deepfake risks listed in the MIT AI Risk Repository.

Rules never change a score, give the same notes for the same video and versions, and attach their
own references. Settings are provisional until they are checked on the test sets.

## The report

The report is filled from a template: every number, note and reference comes from the result. A
language model can write the connecting summary (optional, see [Reports and Ask](guides/report.md)).
It receives the result in words, the notes without numbers and a plain description of your tests,
never the video or frames. Sentences with a number or a conclusion about the video, and references the rules did not
attach, are removed; if anything fails, the template summary is used and the report says so.

## Ask

General questions are answered from a small set of reviewed **knowledge cards**, written from our
survey, the MARS paper and the MOZAIC benchmark, and found with a plain keyword search (no model, no
network). Questions about a result go to the language model, if one is set up, with the result's facts
and the closest cards; the same filter as the report applies. A question asking whether the video is real or fake always
gets the fixed answer that leaves the decision to you.

## Privacy

The video and its frames stay on the machine that runs MARScope. A job and its files are deleted
an hour after last use, or when you leave the result. The only outside calls are the model
downloads from Hugging Face and, if you set one up, the language model.
