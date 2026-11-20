# Reports and Ask

## The report

In the app, **Report** opens the report sheet; on the command line, `--report markdown|html|pdf`
writes it next to the video. It contains:

- a short **summary**, and who wrote it (the template, or a named language model);
- the **scores in context**, per model;
- **What you tried:** the tests you ran in the app (frames hidden, top frames or areas against random,
  occlusion maps, What if edits), with the scores the toolkit measured;
- the **notes**, the **references** they cite (and only those), the **versions** and the disclaimer.

Exports from the sheet:

| Export | What it is |
|---|---|
| Copy | The report as Markdown, on the clipboard |
| Download | PDF, HTML or Markdown, in the app's language |
| Session (JSON) | The full result, your tests, the report text and your guess |
| Evidence card | One PNG: the scores and their place, the trust results, the note on how to read them and the versions |

Tick **Include the current frame** to put the frame you are on in the evidence card, the PDF and the
HTML report. Nothing includes a frame unless you ask, and every file is saved on your machine.

## A language model for the summary (optional)

Without a language model, the summary is a fixed template. To have one write it, add any
OpenAI-compatible endpoint to a `.env` file in the folder you run MARScope from (start from the
repository's `.env.example`: `cp .env.example .env`):

```bash
MARSCOPE_LLM_BASE_URL=https://your-endpoint/v1
MARSCOPE_LLM_API_KEY=...
MARSCOPE_LLM_MODEL=model-name      # exactly as the endpoint lists it (names are case-sensitive)
```

What it receives and what is kept:

- **Sent:** how unusual each score is for a real video (in words), the
  notes with their numbers removed, and your tests in words ("large drop", "little change"). Never
  the video or frames.
- **Kept:** only sentences with no number (in digits or words) and no conclusion about the video, and only references
  the rules attached. Every number in the report comes from the template.
- **If it fails** (no answer within 40 seconds, an error, or prose that breaks the rules), the
  template summary is used and the report says why.

The report sheet shows exactly what was sent, under **Sent to the language model**.

## Ask

The **Learn** tab has **Ask about this result**:

- **Suggested questions** ("Why do the models disagree?", "What do the A1–A5 results say?", "What
  would make this more certain?", "So is it real or fake?") get fixed answers built from the
  result. They work without a language model and give the same answer every time.
- **General questions** ("What is a deepfake?", "What is C2PA?", "What should our team do?") are
  answered from MARScope's **knowledge cards**: short answers written from the cited research papers
  ([survey](https://vectorinstitute.github.io/deepfakes-survey-2026/),
  [MARS](https://vectorinstitute.github.io/mars/), [MOZAIC](https://vectorinstitute.github.io/mozaic/))
  and reviewed. The card's own text is shown, with its references, so these answers work offline and
  never change.
- **Questions about this result** go to the language model with the result's facts, the closest cards as
  background, and the same filter (at most three short sentences, no numbers, no conclusion about the video). Without a
  language model, they get a card only when one answers them closely.
- **A question asking whether the video is real or fake** ("Is this fake?") always gets the fixed answer, which leaves that decision to you, and
  never reaches the language model.
