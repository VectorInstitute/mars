# What it does

MARScope runs trained video deepfake detectors on a video and helps you read what they say. Each
score is placed next to the scores the same model gives real and fake test videos, the evidence
behind it is shown, and you can test what the model relies on. Fixed rules then write notes on how
far to trust the result, with references. The decision about the video stays with you.

---

## A worked example

A 12-second face-swap clip from the Celeb-DF v2 test set, with all three models:

| Model | Probability of fake | Compared with real videos |
|---|---|---|
| MARS | 66% | Very unusual: fewer than 1 in 100 real videos score this high |
| GenD | 89% | Very unusual: fewer than 1 in 100 real videos score this high |
| DFD-HR | 100% | Very unusual: fewer than 1 in 100 real videos score this high |

"66%" on its own says little: MARS gives most real test videos between 7% and 24%, so 66% is far
above its usual range, while for GenD the same 66% would be normal. That is why every score comes
with its place among the model's test videos, and why models are compared by that place, never by
raw score.

The assumption tests then ask, for each model, whether the score breaks when something is removed:

| Test | MARS | GenD | DFD-HR |
|---|---|---|---|
| A1: face outline hidden | Holds | Holds | Holds |
| A4: shared many times | **Breaks** | **Breaks** | **Breaks** |
| A5: fine pixel detail removed | Holds | Partly | Holds |

All three scores fall into the range the models give real videos when the clip is compressed and
resized as if shared many times. A widely shared copy of this clip could pass all three detectors.
That is the kind of thing MARScope is built to show.

---

## What you can do with a result

<div class="grid cards" markdown>

-   __Read the scores in context__

    Each model's probability of fake, how unusual it is for a real video, and how the models
    agree or disagree.

-   __See the evidence__

    MARS's built-in maps of the frames and areas it relies on; occlusion maps for GenD and DFD-HR.

-   __Test it__

    Hide a model's top frames or areas and compare with hiding as many at random. The common test
    is computed in the background, so it is usually ready the moment you press it.

-   __Ask "what if"__

    Change JPEG quality, resolution, blur and noise, or simulate a copy shared many times, and see
    every score move.

-   __Read the notes__

    How far to trust the models (A1–A5), where the video comes from (Content Credentials), and laws
    and risks that may apply in the EU, Canada and the US.

-   __Keep a record__

    A report with what you tried (Markdown, HTML, PDF), an evidence card (PNG) and a session file.
    Ask questions about the result.

</div>

---

## Who it is for

Anyone who wants to understand video deepfake detection rather than trust a single number: people
running workshops and courses, journalists and fact-checkers learning how detectors behave,
researchers comparing detectors, and policy teams who need to know what these tools can and cannot
do.

## What it does not do

- **The decision stays with you.** MARScope reports what each detector measured, with no threshold on
  a score.
- **For learning.** Scores are model outputs and can be wrong. Legal, forensic, journalistic and
  employment decisions, and claims about a person, need qualified experts and other evidence.
- **Video only.** No audio, no images.
- **Not everything.** The models were trained on face videos (FF++) and rendered game footage
  (GTA-V). On videos unlike that, scores are less reliable, and the notes say so.
