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
  "video_confidence": <integer 0-10>,
  "audio_confidence": <integer 0-10>,
  "fake_periods": [[t0, t1], ...],
  "reason": "<one sentence>"
}

Field rules:
- video_fake / audio_fake  : evaluate each stream as a completely independent binary decision
- video_confidence         : 0 = completely uncertain about video, 10 = certain the video prediction is correct
- audio_confidence         : 0 = completely uncertain about audio, 10 = certain the audio prediction is correct
- fake_periods             : time ranges in seconds ``[[t0, t1], ...]`` where manipulation occurs; \
use ``[]`` when the clip is fully real; when fake, list every interval with clear start/end times \
(if manipulation spans the entire clip, return one interval covering the full duration)
- reason                   : the single strongest evidence driving your overall assessment\
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

Important: set audio_fake to false and audio_confidence to 0. \
There is no audio stream, so no audio judgement is possible.

Respond with JSON only.\
"""
