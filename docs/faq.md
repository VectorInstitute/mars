# FAQ

Common questions about what MARScope does and does not do. For the details, see
[How it works](how_it_works.md); for setup, see [Installation](guides/install.md).

---

## About MARScope

### Does MARScope tell me whether a video is fake?

No. It shows what three detectors output, how unusual each score is for that model, what each model
relies on, and how far to trust it. It never says a video is real or fake, and it sets no threshold
on a score. There are three reasons: the scores are uncalibrated, the models often disagree, and a
single answer is easily mistaken for evidence.

### Why does MARScope still leave the decision to me when every model gives a high score?

Agreement makes a reading more consistent, but the three models were trained on the same videos and
can share blind spots. And when fakes are rare, even a detector that is right most of the time is
often wrong about the videos it flags. The **Learn** tab shows this with numbers.

### What does "very unusual for a real video" mean?

Among the real videos the model was tested on (11,098 of them), very few scored this high: "fewer than
1 in 100 real videos score this high on MARS". Each model has its own usual range, so the same score can
be normal for one model and unusual for another, which is why models are compared this way and never
by raw score. It describes the score; what the video shows is for you to judge.

### Why does a model show 100%?

That is the score the model gave, shown as it is. A model can be fully confident and still be
wrong; the assumption tests show whether that confidence survives simple changes to the video.

### Why do the models disagree?

They read different things: MARS looks at the whole frame over time, while GenD and DFD-HR score
frames one by one. Videos unlike the training data are where detectors drift apart. When they
disagree, the next step is to check where the video comes from.

### Which videos work best?

Videos like the training data: face videos (FF++) and rendered footage (GTA-V), at least one full clip long (about 4 seconds),
not heavily compressed. On videos with no face, from a new generator, very short or heavily
compressed, scores are less reliable, and the notes say so (assumption A3).

---

## Privacy

### Does my video leave my machine?

No. The video and its frames stay on the machine that runs MARScope, and a job is deleted an hour
after last use, or when you leave the result. The only outside calls are the model downloads from
Hugging Face and, if you set one up, a language model.

### What does the language model receive?

Only the structured result in words, the notes without their numbers and a plain description of
your tests: never the video or frames. The report sheet shows exactly what was sent. Without a
language model everything still works; the summary then comes from a template.

---

## Setup

### Do I need a GPU?

No, but it helps. With an NVIDIA GPU (Linux or Windows) a 10-second clip takes a few minutes with all
three models; on the CPU it takes much longer. Apple silicon Macs run on the CPU.

### I see "Failed to load libcublasLt.so.12" and the model runs on the CPU

The CUDA libraries are missing. Run `uv sync --extra cuda`, then start again with
`uv run --extra cuda marscope ...`. See [Installation](guides/install.md#the-cuda-extra).

### How do I open the app from my laptop when it runs on a cluster?

Start it with `--host 0.0.0.0` and forward a port over SSH; see
[Web UI and serving](guides/serve.md#on-a-remote-gpu-machine-or-a-cluster).

### Can I use it on Google Colab?

Not tested yet.

---

## Scope

### Can I use the results as evidence, or to make a decision about a person?

No. MARScope is an educational tool. Legal, forensic, journalistic and employment decisions, and
claims about a person, need qualified experts and other evidence. Policy notes are general
information, without legal advice (EU AI Act, Annex III 6(c)).

### Does it check audio or images?

No, video frames only.
