# Long transcript view

The browser keeps the complete ordered segment data for search and copying. Above 160 segments it mounts only the visible document window with overscan, bounded to 160 segment nodes. Small documents retain stable existing nodes during delta updates.

Search uses the complete text index, including offscreen segments. Navigation mounts the matching window before scrolling to it. Copy always includes the full transcript and speaker labels, including segments hidden by the current search. Timecode and width changes invalidate measured row heights. Duplicate snapshots preserve the scroll anchor.

A provisional result is marked as preliminary text in the same document. An SSE epoch reset replaces it when final transcription begins.

Browser regressions use real Chrome through CDP. The fixture replaces the job source, while rendering, filtering, navigation and copying execute unchanged. The long-recording case loads 4000 variable-height segments, scrolls to the last row, searches across offscreen matches and copies all text. The streaming case covers duplicate deltas, reconnect, generations and upload progress.
