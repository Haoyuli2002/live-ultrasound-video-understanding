# Streaming ultrasound QA generation prompt

Generate English question–answer records for the supplied ultrasound video. Choose worthwhile questions yourself from the images. Phrase each one naturally given what has been shown at query time. A reader must be able to verify the decision to wait or answer, and the answer itself, from the visual evidence available at that moment.

## Input and evidence

Use the supplied video ID and timestamps from the original video. If given a clip, preserve its source time offset. Watch the supplied video with its original audio when available. Narration and optional ASR can help locate and understand events. Replay the chosen evidence windows muted to verify that the images alone support the answer. If the narrator gives the answer early, use the first visually sufficient time. If timestamps or visual inputs are unavailable, request the missing input before generating timed records. With partial footage, restrict claims to the reviewed interval. With sparse frames, generate only tasks whose evidence and required review coverage are present.

## Select the event and write the question

1. Identify the case, continuous scan, anatomy, baseline, and observed change. Mark patient changes, replays, slides and edits. For a T-minute video, aim for approximately 4T distinct questions, with T equal to the source duration in seconds divided by 60. Adjust to the visual evidence: repetitive footage can yield fewer, and rich sequences can yield more. Let the footage determine the type mix; a video may yield just one type. Remove paraphrases and recall questions that simply repeat an observation and its answer.
2. Pause at a natural query time. Given only the prefix visible so far, what visual observation, comparison or monitoring request would be informative? Use a uniquely identifiable reference such as “before compression.” Use an exact historical second only for an explicitly bookmarked or time-lookup task. Keep the outcome open.
3. Choose one main type by what the reference answer must deliver. A request to compare a future observation with an earlier one is a temporal comparison even if it starts with “when”.
   - `current_observation`: identify a case-specific finding visible now, including a comparison of two structures within one image.
   - `history_recall`: retrieve a detail from an earlier observation.
   - `temporal_comparison`: compare two corresponding observations. Specify what each contributes. Answer immediately if both are available, or wait for the missing observation.
   - `event_notification`: monitor a predefined, visible condition. Specify what would establish identification, complete visualization, loss, or reacquisition, as appropriate.
4. Record the decision and reason at the query. For WAIT, describe the current visual state and the evidence still missing. For ANSWER, identify the evidence already present. Use English for all questions, answers and reasons. Distinguish screen coordinates from patient anatomy. Keep diagnoses, causal explanations and procedural outcomes within what the images establish.

## Determine when to answer

- Set a natural review horizon independently of whether the target occurs. Starting at the query, inspect every integer second through the first clear answer or that horizon, plus any fractional query or evaluated endpoint. Use the latest source frame at or before each time. Inspect motion and suspected earlier or recurrent events by replaying the interval and checking the boundary at source frame rate when available. Record the sampling and coverage actually inspected.
- Explain the decision at query time using only the prefix available then. A WAIT reason identifies the visible state and the missing evidence. Keep future observations and time remaining out of the reason.
- Use the first clearly supported visual answer on the reviewed sampling grid. For immediate tasks this equals the query time. Skip ambiguous transitions in binary timing judgments. If onset remains unresolved, rewrite the question as an immediate comparison once both observations exist, or omit it.
- A verified untriggered task requires full coverage through the natural observation endpoint with no qualifying event. Use WAIT at query and JSON null for both answer fields. Leave incomplete or unresolved footage for review.
- Select clips and questions on the strength of their visual evidence. Set the query when the request is natural and answer as soon as the evidence is sufficient, whether immediately, a few seconds later or much later. After annotation, you may report waits whose earliest plausible answer is more than 24 seconds after the query for analysis of the current 1 FPS, 24-frame configuration. Evaluate historical dependence separately by checking which answer detail is lost with only the latest 24 frames.

## Check the evidence before accepting a task

- Try to answer using the question and clock alone. Screen for textbook guesses, fixed delays, wording that gives away the historical finding, and repeated question–answer pairs.
- Inspect labels, captions, preset names, diagram text, probe handling, inset appearance and scene changes. For an image-based task, test whether the ultrasound finding still supports the answer with overlays hidden and whether timing can be guessed from external footage alone. Retain meaningful anatomical names. Revise or omit tasks whose essential answer follows from a shortcut.
- Check visibility at the supplied model sampling, resize and crop when available. For memory tasks, test recent frames alone before crediting long-history dependence. State which controls were actually performed, what they showed and what remains unverified. If the essential finding itself cannot be verified visually, omit the task. Model-authored records are candidates pending independent review.

## Output

Produce one `<video_id>.jsonl` file per video, one English candidate question per line as a valid JSON object. If no task qualifies, return zero records. Use source-video seconds and exactly these 11 fields. Keep review notes and sampling coverage in a separate audit report.

| Field | Value |
|---|---|
| `video_id` | Supplied source video ID. |
| `question_id` | Stable question ID unique within that video. |
| `qa_type` | One of the four type strings above. |
| `query_time` | Time when the question can naturally be asked. |
| `question` | English question grounded in the video prefix. |
| `decision_at_query` | `WAIT` or `ANSWER` at query time. |
| `decision_reason_at_query` | English reason at query time. For WAIT, state what is visible and what is missing. For ANSWER, identify the evidence already present. |
| `trigger_criterion` | Observable condition that would justify answering a waiting question, written in English. An immediate descriptive question may use JSON null. |
| `answer_time` | First clearly answerable visual sample, equal to query time for an immediate answer. Use JSON null for a verified untriggered question. |
| `reference_answer` | Concise English answer supported by that time, or JSON null for a verified untriggered question. |
| `evidence_windows` | Shortest sufficient replayable evidence, as inclusive numeric pairs in source seconds. A single sample may use `[[82, 82]]`. For comparisons include both observations in time order; for untriggered questions give the verified observation interval. |

Before emitting each object, check the video and question IDs, the visual support for the query-time reason, and the time bounds. An observed answer must be at or after the query, with all supporting windows ending by answer time. A monitoring question may name the target; keep the outcome and future evidence out of its wording.
