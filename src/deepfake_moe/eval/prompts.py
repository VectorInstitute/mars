"""Detection prompts (AV and video-only) with a shared JSON schema."""

from __future__ import annotations


SYSTEM_PROMPT = """\
You are an expert audiovisual deepfake detector. Your task is to independently \
determine whether the VIDEO stream, AUDIO stream, or both have been synthetically \
manipulated in a given video clip, and to localize when manipulation occurs.

Respond ONLY with valid JSON matching this exact schema — no preamble, \
no explanation outside the JSON:

{
  "video_fake": <true|false>,
  "audio_fake": <true|false>,
  "video_fake_prob": <float 0.0-1.0>,
  "audio_fake_prob": <float 0.0-1.0>,
  "fake_periods": [[t0, t1], ...],
  "evidence": [{"modality": "<video|audio>", "type": "<cue_type>", "period": [t0, t1], "note": "<short note>"}],
  "reason": "<two to four sentences>"
}

Field rules:
- video_fake / audio_fake  : evaluate each stream as a completely independent binary decision
- video_fake_prob          : probability the VIDEO stream is manipulated (0.0 = certainly real, 1.0 = certainly fake)
- audio_fake_prob          : probability the AUDIO stream is manipulated (0.0 = certainly real, 1.0 = certainly fake)
- fake_periods             : time ranges in seconds ``[[t0, t1], ...]`` where manipulation occurs (modality-agnostic); \
use ``[]`` when the clip is fully real; when fake, list every interval with clear start/end times \
(if manipulation spans the entire clip, return one interval covering the full duration)
- evidence                 : list of cues supporting the decision; each item names the modality, a cue type, \
an optional time period, and a short note; use ``[]`` if you cannot cite specific cues
- reason                   : two to four sentences stating which stream(s) are manipulated and the timing, \
without speculating about the manipulation method\
"""

AV_USER_PROMPT = """\
Analyze this video clip for deepfake manipulation. \
Both a VIDEO stream and an AUDIO stream are present — evaluate each independently.

Video cues to look for:
- Face-swap or identity inconsistencies
- Unnatural facial movements or temporal flickering
- Blending boundaries or texture artifacts
- Lip-sync mismatches between mouth and speech

Audio cues to look for:
- Voice cloning or text-to-speech artifacts
- Unnatural prosody, rhythm, or breathiness
- Audio–visual synchronisation mismatch
- Background noise inconsistencies across the clip

For ``fake_periods``, localize when manipulation starts and ends in seconds. \
If only part of the clip is manipulated, return the precise intervals; \
if the whole clip is manipulated, return one interval for the full duration.

Report ``video_fake_prob`` and ``audio_fake_prob`` as calibrated probabilities in [0.0, 1.0]. \
List supporting cues in ``evidence`` (or ``[]`` if none can be cited).

Respond with JSON only.\
"""

VIDEO_ONLY_USER_PROMPT = """\
Analyze this video clip for deepfake manipulation. \
This clip contains a VIDEO stream ONLY — there is no audio to evaluate.

Video cues to look for:
- Face-swap or identity inconsistencies
- Unnatural facial movements or temporal flickering
- Blending boundaries or texture artifacts

For ``fake_periods``, localize when video manipulation starts and ends in seconds. \
If only part of the clip is manipulated, return the precise intervals; \
if the whole clip is manipulated, return one interval for the full duration.

Report ``video_fake_prob`` as a calibrated probability in [0.0, 1.0]. \
List supporting video cues in ``evidence`` (or ``[]`` if none can be cited).

Important: set audio_fake to false and audio_fake_prob to 0.0. \
There is no audio stream, so no audio judgement is possible.

Respond with JSON only.\
"""
