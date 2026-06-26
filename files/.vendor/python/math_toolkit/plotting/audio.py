"""Provide streamed audio state and sampling for playable curve plots."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import TYPE_CHECKING

import numpy as np
import sympy

from ._reactive import Computed, Signal
from .errors import AudioPlaybackError, PlotShapeError, PlotSpecError
from .sampling import compile_numeric_curve
from .specs import CurveView, PLOT_KIND_CURVE

if TYPE_CHECKING:
    from .model import FigureHandle, PlotHandle, PlotNode


@dataclass(frozen=True)
class AudioPlaybackOptions:
    """Describe playback options that affect audio transport and sampling."""

    sample_rate: int = 48_000
    chunk_frames: int = 2_048
    batch_chunks: int = 10
    buffer_seconds: float = 0.25
    gain: float = 1.0
    normalization: bool = True
    normalization_ceiling: float = 0.95
    normalization_attack_seconds: float = 0.005
    normalization_release_seconds: float = 0.75
    phase_match: bool = True
    phase_search_seconds: float = 0.02
    crossfade_seconds: float = 0.005


@dataclass(frozen=True)
class AudioTransportState:
    """Describe Python-owned audio playback transport state."""

    status: str
    time: float
    start: float
    elapsed: float
    emitted_frames: int
    phase_offset: float
    options: AudioPlaybackOptions
    error_message: str | None = None


@dataclass(frozen=True)
class AudioFrontendPosition:
    """Describe the latest frontend-reported audio clock position."""

    session_id: int | None
    elapsed: float | None
    queued_seconds: float | None = None
    requested_chunks: int | None = None
    backend: str | None = None


@dataclass(frozen=True)
class AudioSampleSignature:
    """Describe the mathematical source state needed for one audio sample."""

    expression: object
    domain_symbol: sympy.Symbol
    domain_minimum: float | None
    domain_maximum: float | None
    parameter_symbols: tuple[sympy.Basic, ...]
    parameter_values: tuple[float, ...]


@dataclass(frozen=True)
class AudioTail:
    """Store recently emitted raw function values for continuity matching."""

    positions: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class AudioNormalizationState:
    """Track the private automatic attenuation envelope for one audio node."""

    enabled: bool = True
    gain: float = 1.0


@dataclass(frozen=True)
class AudioChunk:
    """Describe one streamed PCM audio window."""

    session_id: int | None
    source_epoch: int
    sample_rate: int
    start_time: float
    end_time: float
    pcm: np.ndarray
    reached_end: bool = False


@dataclass(frozen=True)
class AudioBatch:
    """Describe one transport batch containing one or more audio chunks."""

    session_id: int | None
    source_epoch: int
    sample_rate: int
    chunk_frames: int
    chunk_count: int
    start_time: float
    end_time: float
    pcm: np.ndarray
    reached_end: bool = False


@dataclass(frozen=True)
class AudioPlaybackState:
    """Describe the current immutable public audio playback state."""

    status: str
    figure_id: int
    node_id: int | None
    plot_name: str | None
    label: str | None
    time: float | None
    elapsed: float | None
    sample_rate: int | None
    gain: float | None
    normalization: bool
    needs_user_activation: bool


@dataclass(frozen=True)
class AudioDebugEvent:
    """Describe one bounded diagnostic event from audio playback."""

    kind: str
    sequence: int
    session_id: int | None
    payload: dict[str, object]


@dataclass(frozen=True)
class AudioTransitionBlend:
    """Describe a short old-to-new signal blend after a matched transition."""

    signature: AudioSampleSignature
    compiled: object
    phase_offset: float
    frames: int


def sound_enabled_for_created_plot(plot_node: PlotNode) -> bool:
    """Return whether a newly created curve plot shows its sound control."""

    return True


def _audio_debug_payloads(
    events: list[dict[str, object]],
    *,
    kind: str | None = None,
) -> list[dict[str, object]]:
    """Return diagnostic payload dictionaries, optionally filtered by kind."""

    payloads: list[dict[str, object]] = []
    for event in events:
        if kind is not None and event.get("kind") != kind:
            continue
        payload = event.get("payload", {})
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _audio_debug_summary(debug: dict[str, object]) -> dict[str, object]:
    """Summarize raw figure diagnostics into click-oriented troubleshooting data."""

    node = debug.get("active_node") or {}
    node = node if isinstance(node, dict) else {}
    figure_events = debug.get("events", [])
    node_events = node.get("events", [])
    figure_events = figure_events if isinstance(figure_events, list) else []
    node_events = node_events if isinstance(node_events, list) else []

    # Keep frontend pressure separate from Python continuity diagnostics. A
    # click caused by queue starvation has a very different fix than a click
    # caused by a discontinuous PCM boundary.
    frontend_payloads = _audio_debug_payloads(figure_events)
    chunk_payloads = _audio_debug_payloads(node_events, kind="chunk")
    transition_payloads = _audio_debug_payloads(node_events, kind="transition")
    underruns = [
        payload for payload in frontend_payloads if payload.get("event") == "underrun"
    ]
    need_data = [
        payload for payload in frontend_payloads if payload.get("event") == "need_data"
    ]
    backend_errors = [
        payload for payload in frontend_payloads if payload.get("event") == "error"
    ]

    # Raw boundary jumps include the normal sample-to-sample slope of the wave.
    # Boundary error removes that expected slope, which is the useful click
    # diagnostic for chunk seams.
    boundary_jumps = [
        abs(float(payload.get("boundary_jump", 0.0)))
        for payload in chunk_payloads
        if payload.get("boundary_jump") is not None
    ]
    boundary_errors = [
        abs(float(payload.get("boundary_error", 0.0)))
        for payload in chunk_payloads
        if payload.get("boundary_error") is not None
    ]
    worst_chunks = sorted(
        (
            payload
            for payload in chunk_payloads
            if payload.get("boundary_error") is not None
        ),
        key=lambda payload: abs(float(payload.get("boundary_error", 0.0))),
        reverse=True,
    )[:5]

    # Transition counts show whether audible clicks line up with delta-x
    # rejection, crossfade fallback, or ordinary accepted phase matches.
    transition_counts: dict[str, int] = {}
    for payload in transition_payloads:
        strategy = str(payload.get("strategy", payload.get("event", "unknown")))
        transition_counts[strategy] = transition_counts.get(strategy, 0) + 1

    return {
        "state": debug.get("state"),
        "frontend": {
            "events": len(figure_events),
            "need_data": len(need_data),
            "underruns": len(underruns),
            "errors": backend_errors[-3:],
            "recent": frontend_payloads[-8:],
        },
        "python": {
            "events": len(node_events),
            "chunks": len(chunk_payloads),
            "transitions": transition_counts,
            "max_boundary_jump": max(boundary_jumps) if boundary_jumps else None,
            "max_boundary_error": max(boundary_errors) if boundary_errors else None,
            "worst_chunks": worst_chunks,
            "recent_transitions": transition_payloads[-8:],
        },
    }


class FigureAudioController:
    """Coordinate active audio selection and output sessions for one figure."""

    def __init__(self, figure: FigureHandle) -> None:
        """Create a controller for a durable figure."""

        self.figure = figure
        self.active_node_signal = Signal(None)
        self.active_output_session_signal = Signal(None)
        self.playback_state = Computed(self._playback_state)
        self._next_session_id = 1
        self._debug_events: list[AudioDebugEvent] = []
        self._debug_sequence = 0
        self._debug_enabled = False

    def select(self, node: AudioNode) -> None:
        """Select an audio node without starting playback."""

        active = self.active_node_signal()
        if active is node:
            return
        if active is not None:
            active.stop()
        self.active_node_signal.set(node)
        self.active_output_session_signal.set(None)

    def play(self, node: AudioNode, options: AudioPlaybackOptions, start: float | None) -> None:
        """Select an audio node, start it, and allocate an output session id."""

        active = self.active_node_signal()
        if active is not None and active is not node:
            active.stop()
        self.active_node_signal.set(node)
        node.start(options, start=start)
        self.active_output_session_signal.set(self._new_session_id())

    def stop(self) -> None:
        """Stop the active audio node and retire the frontend output session."""

        active = self.active_node_signal()
        if active is not None:
            active.stop()
        self.figure._send_audio_output_command({"type": "stop"})
        self.active_output_session_signal.set(None)

    def pause(self) -> None:
        """Pause the active audio node if one is selected."""

        active = self.active_node_signal()
        if active is not None:
            active.pause()
        session_id = self.active_output_session_signal()
        self.active_output_session_signal.set(None)
        if session_id is not None:
            self.figure._send_audio_output_command(
                {"type": "pause", "session_id": session_id}
            )

    def resume(self) -> None:
        """Resume the selected audio node without changing the selection."""

        active = self.active_node_signal()
        if active is None:
            return
        active.resume()
        if self.active_output_session_signal() is None:
            self.active_output_session_signal.set(self._new_session_id())
        self.start_frontend_output()

    def start_frontend_output(self) -> None:
        """Ask the active browser output adapter to start streaming audio."""

        active = self.active_node_signal()
        session_id = self.active_output_session_signal()
        if active is None or session_id is None:
            return
        state = active.transport_signal()
        options = state.options
        self.figure._send_audio_output_command(
            {
                "type": "start",
                "session_id": session_id,
                "sample_rate": options.sample_rate,
                "chunk_frames": options.chunk_frames,
                "batch_chunks": options.batch_chunks,
                "buffer_seconds": options.buffer_seconds,
                "debug": self._debug_enabled,
            }
        )

    def request_chunk(self, session_id: int, frame_count: int | None = None) -> AudioChunk | None:
        """Return a PCM chunk for the active session or ignore stale requests."""

        if session_id != self.active_output_session_signal():
            return None
        active = self.active_node_signal()
        if active is None:
            return None
        try:
            chunk = active.sample_next_window(frame_count)
        except Exception as exc:
            active.fail(str(exc))
            self.active_output_session_signal.set(None)
            return None
        if chunk.reached_end:
            self.active_output_session_signal.set(None)
        return replace(chunk, session_id=session_id)

    def request_batch(
        self,
        session_id: int,
        chunk_count: int | None = None,
    ) -> AudioBatch | None:
        """Return a transport batch made from consecutive playback chunks."""

        if session_id != self.active_output_session_signal():
            return None
        active = self.active_node_signal()
        if active is None:
            return None
        state = active.transport_signal()
        options = state.options
        count = (
            options.batch_chunks
            if chunk_count is None
            else _positive_int(chunk_count, "batch chunk count")
        )

        chunks: list[AudioChunk] = []
        try:
            for _ in range(count):
                chunk = active.sample_next_window(options.chunk_frames)
                chunks.append(chunk)
                if chunk.reached_end:
                    break
        except Exception as exc:
            active.fail(str(exc))
            self.active_output_session_signal.set(None)
            return None
        if not chunks:
            return None
        if chunks[-1].reached_end:
            self.active_output_session_signal.set(None)
        pcm = np.concatenate([chunk.pcm for chunk in chunks]).astype(
            np.float32,
            copy=False,
        )
        return AudioBatch(
            session_id=session_id,
            source_epoch=chunks[-1].source_epoch,
            sample_rate=chunks[-1].sample_rate,
            chunk_frames=options.chunk_frames,
            chunk_count=len(chunks),
            start_time=chunks[0].start_time,
            end_time=chunks[-1].end_time,
            pcm=pcm,
            reached_end=chunks[-1].reached_end,
        )

    def handle_frontend_message(self, content: dict[str, object], buffers: object) -> None:
        """Handle one custom message from the browser audio adapter."""

        message_type = content.get("type")
        session_id = content.get("session_id")
        if message_type == "chunk_request":
            if not isinstance(session_id, int):
                return
            frame_count = content.get("frame_count")
            chunk = self.request_chunk(
                session_id,
                frame_count if isinstance(frame_count, int) else None,
            )
            if chunk is None:
                self.figure._send_audio_output_command(
                    {"type": "stale", "session_id": session_id}
                )
                return
            self.figure._send_audio_output_chunk(chunk)
            return
        if message_type == "batch_request":
            if not isinstance(session_id, int):
                return
            chunk_count = content.get("chunk_count")
            batch = self.request_batch(
                session_id,
                chunk_count if isinstance(chunk_count, int) else None,
            )
            if batch is None:
                self.figure._send_audio_output_command(
                    {"type": "stale", "session_id": session_id}
                )
                return
            self.figure._send_audio_output_batch(batch)
            return

        if not isinstance(session_id, int):
            return
        if session_id != self.active_output_session_signal():
            return

        if message_type == "position":
            elapsed = content.get("elapsed")
            if isinstance(elapsed, int | float):
                self.report_frontend_position(session_id, content)
            return
        if message_type == "diagnostic":
            self.record_debug_event(
                "frontend",
                session_id,
                {
                    key: value
                    for key, value in content.items()
                    if key not in {"type", "session_id"}
                },
            )
            return
        if message_type == "needs_user_activation":
            active = self.active_node_signal()
            if active is not None:
                active.needs_user_activation()
            self.active_output_session_signal.set(None)
            return
        if message_type == "error":
            active = self.active_node_signal()
            if active is not None:
                active.fail(str(content.get("message", "Audio output failed.")))
            self.active_output_session_signal.set(None)

    def report_frontend_position(
        self,
        session_id: int,
        content: dict[str, object],
    ) -> None:
        """Accept a frontend clock report only from the active output session."""

        if session_id != self.active_output_session_signal():
            return
        active = self.active_node_signal()
        if active is not None:
            elapsed = content.get("elapsed")
            queued_seconds = content.get("queued_seconds")
            requested_chunks = content.get("requested_chunks")
            backend = content.get("backend")
            active.frontend_position_signal.set(
                AudioFrontendPosition(
                    session_id=session_id,
                    elapsed=float(elapsed) if isinstance(elapsed, int | float) else None,
                    queued_seconds=(
                        float(queued_seconds)
                        if isinstance(queued_seconds, int | float)
                        else None
                    ),
                    requested_chunks=(
                        int(requested_chunks)
                        if isinstance(requested_chunks, int)
                        else None
                    ),
                    backend=str(backend) if isinstance(backend, str) else None,
                )
            )
            self.record_debug_event(
                "position",
                session_id,
                {
                    key: value
                    for key, value in content.items()
                    if key not in {"type", "session_id"}
                },
            )

    def record_debug_event(
        self,
        kind: str,
        session_id: int | None,
        payload: dict[str, object],
    ) -> None:
        """Append one bounded audio diagnostic event."""

        if not self._debug_enabled:
            return
        self._debug_sequence += 1
        self._debug_events.append(
            AudioDebugEvent(
                kind=kind,
                sequence=self._debug_sequence,
                session_id=session_id,
                payload=dict(payload),
            )
        )
        del self._debug_events[:-200]

    def debug(self) -> dict[str, object]:
        """Return recent audio diagnostics as plain copyable data."""

        active = self.active_node_signal()
        active_debug = active.debug() if active is not None else None
        return {
            "figure_id": self.figure.id,
            "active_session_id": self.active_output_session_signal(),
            "state": asdict(self.state()),
            "active_node": active_debug,
            "events": [asdict(event) for event in self._debug_events],
        }

    def clear_debug(self) -> None:
        """Clear accumulated figure and active-node audio diagnostics."""

        self._debug_events.clear()
        active = self.active_node_signal()
        if active is not None:
            active.clear_debug()

    def set_debug_enabled(self, enabled: bool) -> None:
        """Enable or disable browser and Python audio diagnostics."""

        self._debug_enabled = bool(enabled)
        self.figure._send_audio_output_command(
            {"type": "debug", "enabled": self._debug_enabled}
        )

    def state(self) -> AudioPlaybackState:
        """Return the current immutable playback state record."""

        return self.playback_state()

    def _playback_state(self) -> AudioPlaybackState:
        """Return the figure-level state for the active node, if any."""

        active = self.active_node_signal()
        if active is None:
            return AudioPlaybackState(
                status="stopped",
                figure_id=self.figure.id,
                node_id=None,
                plot_name=None,
                label=None,
                time=None,
                elapsed=None,
                sample_rate=None,
                gain=None,
                normalization=False,
                needs_user_activation=False,
            )
        return active.playback_state()

    def _new_session_id(self) -> int:
        """Return a fresh monotonically increasing output session id."""

        session_id = self._next_session_id
        self._next_session_id += 1
        return session_id


class AudioNode:
    """Own audio sampling state for one ordinary curve plot node."""

    _ids = 0

    def __init__(self, figure: FigureHandle, plot_node: PlotNode) -> None:
        """Create an audio node attached to one curve plot node."""

        if plot_node.kind != PLOT_KIND_CURVE:
            raise AudioPlaybackError("Only ordinary plot(...) curves can be played as sound.")
        AudioNode._ids += 1
        self.id = AudioNode._ids
        self.figure = figure
        self.plot_node = plot_node
        self.sound_enabled_signal = Signal(sound_enabled_for_created_plot(plot_node))
        self.normalization_signal = Signal(True)
        self.transport_signal = Signal(_stopped_transport(self._default_start()))
        self.frontend_position_signal = Signal(AudioFrontendPosition(None, None))
        self.audio_sample_signature = Computed(
            self._audio_sample_signature,
            equal=_semantic_equal,
        )
        self.playback_state = Computed(self._playback_state, equal=_semantic_equal)
        self._compiled_cache: dict[tuple[object, ...], object] = {}
        self._last_signature: AudioSampleSignature | None = None
        self._tail: AudioTail | None = None
        self._disposed = False
        self._last_transition: str | None = None
        self._phase_match_attempts = 0
        self._crossfade_uses = 0
        self._source_epoch = 0
        self._transition_blend: AudioTransitionBlend | None = None
        self._normalization_state = AudioNormalizationState()
        self._debug_events: list[AudioDebugEvent] = []
        self._debug_sequence = 0

    @property
    def last_transition(self) -> str | None:
        """Return the transition strategy used by the previous emitted chunk."""

        return self._last_transition

    @property
    def phase_match_attempts(self) -> int:
        """Return how many delta-x matching attempts this node has made."""

        return self._phase_match_attempts

    @property
    def crossfade_uses(self) -> int:
        """Return how many emitted chunks used crossfade fallback."""

        return self._crossfade_uses

    def start(self, options: AudioPlaybackOptions, *, start: float | None) -> None:
        """Start playback from the requested or stored domain time."""

        start_time = self._normalized_start(start)
        self._last_signature = None
        self._tail = None
        self._last_transition = None
        self._source_epoch = 0
        self._transition_blend = None
        self._normalization_state = AudioNormalizationState(
            enabled=options.normalization,
        )
        self._record_debug_event(
            "node_start",
            {
                "start": start_time,
                "sample_rate": options.sample_rate,
                "chunk_frames": options.chunk_frames,
                "batch_chunks": options.batch_chunks,
                "buffer_seconds": options.buffer_seconds,
                "gain": options.gain,
                "normalization": options.normalization,
                "normalization_ceiling": options.normalization_ceiling,
                "phase_match": options.phase_match,
                "phase_search_seconds": options.phase_search_seconds,
                "crossfade_seconds": options.crossfade_seconds,
            },
        )
        self.transport_signal.set(
            AudioTransportState(
                status="playing",
                time=start_time,
                start=start_time,
                elapsed=0.0,
                emitted_frames=0,
                phase_offset=0.0,
                options=options,
            )
        )

    def stop(self) -> None:
        """Stop playback while preserving the current stored domain time."""

        state = self.transport_signal()
        if state.status == "stopped":
            return
        self._record_debug_event("node_stop", {"time": state.time, "elapsed": state.elapsed})
        self.transport_signal.set(replace(state, status="stopped"))

    def pause(self) -> None:
        """Pause playback without changing the current domain time."""

        state = self.transport_signal()
        if state.status in {"stopped", "paused"}:
            return
        self._record_debug_event("node_pause", {"time": state.time, "elapsed": state.elapsed})
        self.transport_signal.set(replace(state, status="paused"))

    def resume(self) -> None:
        """Resume playback from the current stored domain time."""

        state = self.transport_signal()
        if state.status == "playing":
            return
        self._last_signature = None
        self._tail = None
        self._transition_blend = None
        self._normalization_state = AudioNormalizationState(
            enabled=state.options.normalization,
        )
        self._record_debug_event("node_resume", {"time": state.time})
        self.transport_signal.set(
            replace(
                state,
                status="playing",
                start=state.time,
                elapsed=0.0,
                emitted_frames=0,
                phase_offset=0.0,
                error_message=None,
            )
        )

    def seek(self, value: float) -> None:
        """Set the current domain playback time."""

        target = _finite_float(value, "Audio time")
        state = self.transport_signal()
        self._last_signature = None
        self._tail = None
        self._transition_blend = None
        self._normalization_state = AudioNormalizationState(
            enabled=state.options.normalization,
        )
        self._record_debug_event("node_seek", {"from": state.time, "to": target})
        self.transport_signal.set(
            replace(
                state,
                time=target,
                start=target,
                elapsed=0.0,
                emitted_frames=0,
                phase_offset=0.0,
            )
        )

    def reset(self) -> None:
        """Seek this audio node to domain time zero."""

        self.seek(0.0)

    def fail(self, message: str) -> None:
        """Move the node into an error state with a queryable message."""

        self._record_debug_event("node_error", {"message": message})
        self.transport_signal.set(
            replace(self.transport_signal(), status="error", error_message=message)
        )

    def needs_user_activation(self) -> None:
        """Report that the browser blocked audio startup without a user gesture."""

        self.transport_signal.set(
            replace(
                self.transport_signal(),
                status="needs_user_activation",
                error_message=(
                    "Browser audio playback needs a user activation before "
                    "the speaker backend can start."
                ),
            )
        )

    def dispose(self) -> None:
        """Stop playback and release cached audio resources."""

        self._record_debug_event("node_dispose", {"time": self.transport_signal().time})
        self.stop()
        self._disposed = True
        self._compiled_cache.clear()
        self._tail = None
        self._transition_blend = None
        self._normalization_state = AudioNormalizationState()

    def set_normalization(self, enabled: bool) -> None:
        """Set the persistent automatic normalization toggle."""

        if not isinstance(enabled, bool):
            raise AudioPlaybackError("Curve sound normalization must be True or False.")
        self.normalization_signal.set(enabled)

        # Keep the active stream's immutable options aligned with the public
        # toggle so the next chunk and public state observe the same setting.
        state = self.transport_signal()
        if state.options.normalization != enabled:
            self.transport_signal.set(
                replace(state, options=replace(state.options, normalization=enabled))
            )

    def sample_next_window(self, frame_count: int | None = None) -> AudioChunk:
        """Sample and return the next consecutive PCM window."""

        if self._disposed:
            raise AudioPlaybackError("Cannot sample audio from a disposed plot.")

        state = self.transport_signal()
        options = state.options
        frames = options.chunk_frames if frame_count is None else _positive_int(frame_count, "chunk frame count")
        if state.status != "playing":
            return AudioChunk(
                session_id=None,
                source_epoch=self._source_epoch,
                sample_rate=options.sample_rate,
                start_time=state.time,
                end_time=state.time,
                pcm=np.zeros(frames, dtype=np.float32),
            )

        signature = self.audio_sample_signature()
        if self._last_signature is not None and not _semantic_equal(
            signature,
            self._last_signature,
        ):
            self._source_epoch += 1
        compiled = self._compiled_for_signature(signature)
        transition = self._prepare_transition(signature, compiled, state)
        state = self.transport_signal()

        positions = state.time + np.arange(frames, dtype=float) / options.sample_rate
        positions = positions + state.phase_offset
        max_value = signature.domain_maximum
        reached_end = False
        if max_value is not None:
            in_domain = positions <= max_value
            if not bool(np.all(in_domain)):
                reached_end = True
                valid_count = int(np.count_nonzero(in_domain))
                positions = positions[:valid_count]

        raw_values = self._evaluate(compiled, signature, positions)
        raw_values = self._apply_transition_blend(raw_values, positions, state)
        if transition == "crossfade":
            raw_values = self._crossfade_values(raw_values, positions, state)

        finite_values = np.where(np.isfinite(raw_values), raw_values, 0.0)
        normalized_values, normalization_debug = self._apply_auto_normalization(
            finite_values,
            options,
        )
        pcm_values = normalized_values * options.gain
        pcm = pcm_values.astype(np.float32, copy=False)
        if pcm.shape[0] < frames:
            pcm = np.pad(pcm, (0, frames - pcm.shape[0])).astype(np.float32, copy=False)

        emitted = positions.shape[0]
        next_time = state.time + emitted / options.sample_rate

        # Natural finite-domain completion should behave like a finished clip:
        # the node is stopped, but the stored playhead is ready for another
        # click rather than parked just beyond the domain end.
        if reached_end:
            status = "stopped"
            stored_time = self._default_start()
            elapsed = 0.0
            emitted_frames = 0
        else:
            status = "playing"
            stored_time = next_time
            elapsed = state.elapsed + emitted / options.sample_rate
            emitted_frames = state.emitted_frames + emitted
        next_state = replace(
            self.transport_signal(),
            status=status,
            time=stored_time,
            elapsed=elapsed,
            emitted_frames=emitted_frames,
        )
        self.transport_signal.set(next_state)
        previous_tail = self._tail
        self._tail = _tail_from_samples(positions, finite_values, options.sample_rate)
        self._last_signature = signature
        self._last_transition = transition
        if self.figure._audio_controller._debug_enabled:
            boundary_jump = None
            boundary_expected = None
            boundary_error = None
            if (
                previous_tail is not None
                and previous_tail.values.size > 0
                and finite_values.size > 0
            ):
                previous_value = float(previous_tail.values[-1])
                previous_slope = (
                    float(previous_tail.values[-1] - previous_tail.values[-2])
                    if previous_tail.values.size >= 2
                    else 0.0
                )
                boundary_expected = previous_value + previous_slope
                boundary_jump = float(abs(float(finite_values[0]) - previous_value))
                boundary_error = float(abs(float(finite_values[0]) - boundary_expected))
            self._record_debug_event(
                "chunk",
                {
                    "status": next_state.status,
                    "transition": transition or "none",
                    "frames_requested": frames,
                    "frames_emitted": int(emitted),
                    "start_time": state.time,
                    "end_time": next_time,
                    "phase_offset": state.phase_offset,
                    "source_epoch": self._source_epoch,
                    "boundary_jump": boundary_jump,
                    "boundary_expected": boundary_expected,
                    "boundary_error": boundary_error,
                    "min": float(np.min(finite_values)) if finite_values.size else None,
                    "max": float(np.max(finite_values)) if finite_values.size else None,
                    "rms": (
                        float(np.sqrt(np.mean(np.square(finite_values))))
                        if finite_values.size
                        else None
                    ),
                    **normalization_debug,
                },
            )

        return AudioChunk(
            session_id=None,
            source_epoch=self._source_epoch,
            sample_rate=options.sample_rate,
            start_time=state.time,
            end_time=next_time,
            pcm=pcm,
            reached_end=reached_end,
        )

    def _audio_sample_signature(self) -> AudioSampleSignature:
        """Read the plot's mathematical source state for audio sampling."""

        view = self.plot_node.view_signal()
        if not isinstance(view, CurveView):
            raise AudioPlaybackError("Only ordinary plot(...) curves can be played as sound.")
        symbols = self.plot_node.parameter_symbols_signal()
        return AudioSampleSignature(
            expression=self.plot_node.expression_signal(),
            domain_symbol=view.x_domain.symbol,
            domain_minimum=view.x_domain.minimum,
            domain_maximum=view.x_domain.maximum,
            parameter_symbols=symbols,
            parameter_values=tuple(
                self.plot_node.parameters[symbol].value_signal()
                for symbol in symbols
            ),
        )

    def _playback_state(self) -> AudioPlaybackState:
        """Return this node's current immutable playback state."""

        state = self.transport_signal()
        frontend_position = self.frontend_position_signal()
        elapsed = frontend_position.elapsed if frontend_position.elapsed is not None else state.elapsed
        return AudioPlaybackState(
            status=state.status,
            figure_id=self.figure.id,
            node_id=self.id,
            plot_name=self.plot_node.name,
            label=self.plot_node.label_signal(),
            time=state.time,
            elapsed=elapsed,
            sample_rate=state.options.sample_rate,
            gain=state.options.gain,
            normalization=state.options.normalization,
            needs_user_activation=state.status == "needs_user_activation",
        )

    def _compiled_for_signature(self, signature: AudioSampleSignature) -> object:
        """Return a cached numeric callable for the audio sample signature."""

        key = (
            signature.expression,
            signature.domain_symbol,
            signature.parameter_symbols,
        )
        compiled = self._compiled_cache.get(key)
        if compiled is None:
            compiled = compile_numeric_curve(
                signature.expression,
                signature.domain_symbol,
                signature.parameter_symbols,
            )
            self._compiled_cache[key] = compiled
        return compiled

    def _prepare_transition(
        self,
        signature: AudioSampleSignature,
        compiled: object,
        state: AudioTransportState,
    ) -> str | None:
        """Apply phase matching when the source signature changed midstream."""

        if self._last_signature is None or self._last_signature == signature:
            return None
        if state.emitted_frames <= 0 or self._tail is None:
            return None

        options = state.options
        self._phase_match_attempts += 1
        if options.phase_match:
            delta, metrics = self._find_phase_delta(compiled, signature, options, state)
            blend_frames = (
                max(1, int(options.crossfade_seconds * options.sample_rate))
                if delta is not None and options.crossfade_seconds > 0
                else 0
            )
            self._record_debug_event(
                "transition",
                {
                    "strategy": "delta-x" if delta is not None else "delta-x-rejected",
                    "blend_frames": blend_frames,
                    **metrics,
                },
            )
            if delta is not None:
                old_signature = self._last_signature
                if old_signature is not None and options.crossfade_seconds > 0:
                    self._transition_blend = AudioTransitionBlend(
                        signature=old_signature,
                        compiled=self._compiled_for_signature(old_signature),
                        phase_offset=state.phase_offset,
                        frames=blend_frames,
                    )
                self.transport_signal.set(
                    replace(state, phase_offset=state.phase_offset + delta)
                )
                return "delta-x"
        self._crossfade_uses += 1
        self._record_debug_event(
            "transition",
            {
                "strategy": "crossfade",
                "phase_match": options.phase_match,
                "crossfade_seconds": options.crossfade_seconds,
            },
        )
        return "crossfade"

    def _find_phase_delta(
        self,
        compiled: object,
        signature: AudioSampleSignature,
        options: AudioPlaybackOptions,
        state: AudioTransportState,
    ) -> tuple[float | None, dict[str, object]]:
        """Return a local x shift and diagnostics for the sampled boundary."""

        tail = self._tail
        if tail is None or tail.positions.size == 0:
            return None, {"reason": "missing_tail"}

        search = float(options.phase_search_seconds)
        frame_step = 1.0 / options.sample_rate
        coarse_candidates = np.linspace(-search, search, 81)
        best_score = math.inf
        best_delta: float | None = None
        old_values = tail.values
        old_slope = old_values[-1] - old_values[-2] if old_values.size >= 2 else 0.0
        first_position = state.time + state.phase_offset
        boundary_frames = min(max(2, int(0.002 * options.sample_rate)), 96)
        boundary_positions = first_position + np.arange(boundary_frames, dtype=float) * frame_step
        expected_first = old_values[-1] + old_slope
        recent_count = min(old_values.size, max(4, int(0.004 * options.sample_rate)))
        recent_positions = tail.positions[-recent_count:]
        recent_values = old_values[-recent_count:]
        best_delta, best_score = self._score_phase_candidates(
            coarse_candidates,
            compiled,
            signature,
            boundary_positions,
            recent_positions,
            recent_values,
            expected_first,
            old_slope,
        )

        if best_delta is None:
            return None, {
                "reason": "no_finite_candidate",
                "search_seconds": search,
                "candidates": int(coarse_candidates.size),
            }

        # Refine around the best coarse hit with a small bounded grid. The
        # browser clock cannot wait for a dense search during slider motion;
        # continuity is more important than sub-frame optimality here.
        coarse_step = (
            float(coarse_candidates[1] - coarse_candidates[0])
            if coarse_candidates.size > 1
            else search
        )
        refine_radius = max(coarse_step, frame_step)
        refine_count = min(
            65,
            max(3, int(math.ceil((2 * refine_radius) / frame_step)) + 1),
        )
        refine_candidates = np.linspace(
            best_delta - refine_radius,
            best_delta + refine_radius,
            refine_count,
        )
        refine_candidates = refine_candidates[
            (refine_candidates >= -search) & (refine_candidates <= search)
        ]
        refined_delta, refined_score = self._score_phase_candidates(
            refine_candidates,
            compiled,
            signature,
            boundary_positions,
            recent_positions,
            recent_values,
            expected_first,
            old_slope,
        )
        if refined_delta is not None and refined_score <= best_score:
            best_delta = refined_delta
            best_score = refined_score

        # Finally polish the chosen delta using only the first boundary sample.
        # This is cheap enough for slider motion and directly targets the seam
        # that produces clicks, without rerunning the full tail score.
        best_delta = self._polish_boundary_delta(
            best_delta,
            frame_step,
            compiled,
            signature,
            first_position,
            expected_first,
        )
        boundary_values = self._evaluate(
            compiled,
            signature,
            boundary_positions + best_delta,
        )
        boundary_error = (
            float(abs(boundary_values[0] - expected_first))
            if boundary_values.size > 0 and np.isfinite(boundary_values[0])
            else math.inf
        )
        boundary_value_error = (
            float(abs(boundary_values[0] - old_values[-1]))
            if boundary_values.size > 0 and np.isfinite(boundary_values[0])
            else math.inf
        )
        threshold = max(0.2, float(np.nanstd(old_values)) * 1.5)
        value_threshold = max(0.03, float(np.nanstd(old_values)) * 0.08)
        boundary_threshold = max(0.005, float(np.nanstd(old_values)) * 0.02)
        if (
            best_score > threshold
            and boundary_value_error > value_threshold
            and boundary_error > boundary_threshold
        ):
            return None, {
                "reason": "score_above_threshold",
                "best_delta": best_delta,
                "best_score": best_score,
                "threshold": threshold,
                "boundary_error": boundary_error,
                "boundary_value_error": boundary_value_error,
                "value_threshold": value_threshold,
                "boundary_threshold": boundary_threshold,
                "search_seconds": search,
                "coarse_candidates": int(coarse_candidates.size),
                "refine_candidates": int(refine_candidates.size),
                "old_last": float(old_values[-1]),
                "old_slope": float(old_slope),
            }
        return best_delta, {
            "best_delta": best_delta,
            "best_score": best_score,
            "threshold": threshold,
            "boundary_error": boundary_error,
            "boundary_value_error": boundary_value_error,
            "value_threshold": value_threshold,
            "boundary_threshold": boundary_threshold,
            "search_seconds": search,
            "coarse_candidates": int(coarse_candidates.size),
            "refine_candidates": int(refine_candidates.size),
            "old_last": float(old_values[-1]),
            "old_slope": float(old_slope),
        }

    def _polish_boundary_delta(
        self,
        delta: float,
        step: float,
        compiled: object,
        signature: AudioSampleSignature,
        first_position: float,
        expected_first: float,
    ) -> float:
        """Locally reduce first-sample mismatch around an accepted phase delta."""

        best_delta = float(delta)
        best_error = self._boundary_value_error(
            best_delta,
            compiled,
            signature,
            first_position,
            expected_first,
        )

        # Search a tiny neighborhood with a shrinking step. This keeps the
        # phase matcher responsive while removing most one-frame quantization at
        # the audible chunk boundary.
        local_step = float(step)
        for _ in range(8):
            candidates = (best_delta - local_step, best_delta + local_step)
            for candidate in candidates:
                error = self._boundary_value_error(
                    candidate,
                    compiled,
                    signature,
                    first_position,
                    expected_first,
                )
                if error < best_error:
                    best_delta = float(candidate)
                    best_error = error
            local_step *= 0.5
        return best_delta

    def _boundary_value_error(
        self,
        delta: float,
        compiled: object,
        signature: AudioSampleSignature,
        first_position: float,
        expected_first: float,
    ) -> float:
        """Return first-sample continuity error for one phase delta."""

        try:
            values = self._evaluate(
                compiled,
                signature,
                np.asarray([first_position + delta], dtype=float),
            )
        except PlotShapeError:
            return math.inf
        if values.size == 0 or not np.isfinite(values[0]):
            return math.inf
        return float(abs(values[0] - expected_first))

    def _score_phase_candidates(
        self,
        candidates: np.ndarray,
        compiled: object,
        signature: AudioSampleSignature,
        boundary_positions: np.ndarray,
        recent_positions: np.ndarray,
        recent_values: np.ndarray,
        expected_first: float,
        old_slope: float,
    ) -> tuple[float | None, float]:
        """Return the best candidate delta and continuity score."""

        best_score = math.inf
        best_delta: float | None = None
        for delta in candidates:
            try:
                boundary_values = self._evaluate(
                    compiled,
                    signature,
                    boundary_positions + delta,
                )
            except PlotShapeError:
                continue
            finite_boundary = np.isfinite(boundary_values)
            if not bool(np.all(finite_boundary[:2])):
                continue
            first_error = abs(boundary_values[0] - expected_first)
            slope_error = abs((boundary_values[1] - boundary_values[0]) - old_slope)

            try:
                recent_candidate = self._evaluate(
                    compiled,
                    signature,
                    recent_positions + delta,
                )
            except PlotShapeError:
                recent_error = math.inf
            else:
                finite_recent = np.isfinite(recent_candidate)
                if bool(np.any(finite_recent)):
                    recent_error = float(
                        np.nanmean(
                            np.square(
                                recent_candidate[finite_recent]
                                - recent_values[finite_recent]
                            )
                        )
                    )
                else:
                    recent_error = math.inf

            score = float(first_error * 30 + slope_error * 12 + recent_error * 0.25)
            if score < best_score:
                best_score = score
                best_delta = float(delta)
        return best_delta, best_score

    def _apply_transition_blend(
        self,
        values: np.ndarray,
        positions: np.ndarray,
        state: AudioTransportState,
    ) -> np.ndarray:
        """Smooth the leading samples after a phase-matched source change."""

        blend = self._transition_blend
        self._transition_blend = None
        if blend is None or values.size == 0 or positions.size == 0:
            return values

        # Evaluate only the short leading window from the previous source. This
        # de-zippers abrupt frequency changes while keeping the stream
        # generated in small chunks.
        fade_frames = min(blend.frames, values.size)
        old_positions = (
            state.time
            + np.arange(fade_frames, dtype=float) / state.options.sample_rate
            + blend.phase_offset
        )
        try:
            old_values = self._evaluate(blend.compiled, blend.signature, old_positions)
        except PlotShapeError:
            return values
        if old_values.size == 0:
            return values

        # A raised-cosine fade keeps both ends flat enough to avoid adding a new
        # sharp corner while the old signal hands off to the phase-matched one.
        weights = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, fade_frames))
        blended = values.copy()
        blended[:fade_frames] = (
            old_values[:fade_frames] * (1.0 - weights)
            + values[:fade_frames] * weights
        )
        return blended

    def _crossfade_values(
        self,
        values: np.ndarray,
        positions: np.ndarray,
        state: AudioTransportState,
    ) -> np.ndarray:
        """Blend the leading samples from the previous tail into new values."""

        tail = self._tail
        if tail is None or tail.values.size == 0 or values.size == 0:
            return values
        if state.options.crossfade_seconds <= 0:
            return values
        fade_frames = max(1, int(state.options.crossfade_seconds * state.options.sample_rate))
        fade_frames = min(fade_frames, values.size)
        slope = tail.values[-1] - tail.values[-2] if tail.values.size >= 2 else 0.0
        old_start = tail.values[-1] + slope * np.arange(1, fade_frames + 1, dtype=float)
        weights = np.linspace(0.0, 1.0, fade_frames)
        blended = values.copy()
        blended[:fade_frames] = old_start * (1.0 - weights) + values[:fade_frames] * weights
        return blended

    def _evaluate(
        self,
        compiled: object,
        signature: AudioSampleSignature,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Evaluate an audio numeric callable as a real one-dimensional array."""

        if positions.size == 0:
            return np.empty(0, dtype=float)
        try:
            raw_values = compiled(positions, *signature.parameter_values)
        except Exception as exc:
            raise PlotShapeError(
                "Audio playback supports real scalar curve expressions only."
            ) from exc
        return _real_array_like(
            raw_values,
            positions.shape,
            message="Audio playback supports real scalar curve expressions only.",
        )

    def _default_start(self) -> float:
        """Return the default domain start for this curve."""

        view = self.plot_node.view_signal()
        if isinstance(view, CurveView) and view.x_domain.minimum is not None:
            return float(view.x_domain.minimum)
        return 0.0

    def _normalized_start(self, start: float | None) -> float:
        """Return the explicit start, stored time, or domain default."""

        if start is not None:
            return _finite_float(start, "Audio start")
        return self._default_start()

    def _apply_auto_normalization(
        self,
        values: np.ndarray,
        options: AudioPlaybackOptions,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Apply automatic attenuation and return debug metadata."""

        peak = float(np.max(np.abs(values))) if values.size else 0.0
        if not options.normalization:
            self._normalization_state = AudioNormalizationState(enabled=False, gain=1.0)
            return values, {
                "normalization": False,
                "normalization_peak": peak,
                "normalization_target_gain": 1.0,
                "normalization_gain_start": 1.0,
                "normalization_gain_end": 1.0,
                "normalization_limited": False,
            }

        target = 1.0 if peak <= 0.0 else min(1.0, options.normalization_ceiling / peak)
        gain_start = self._normalization_state.gain

        # Clamp loud chunks immediately, then recover over time on chunks whose
        # peak leaves headroom. The update is chunk-rate but derived from the
        # sample-rate release time so small and large chunks recover similarly.
        if target < gain_start:
            gain_end = target
        else:
            frames = max(1, int(values.shape[0]))
            release_alpha = 1.0 - math.exp(
                -frames / (options.normalization_release_seconds * options.sample_rate)
            )
            gain_end = gain_start + release_alpha * (target - gain_start)
        gain_end = min(1.0, max(0.0, float(gain_end)))
        if gain_end <= 0.0:
            gain_end = min(1.0, options.normalization_ceiling)
        self._normalization_state = AudioNormalizationState(enabled=True, gain=gain_end)

        return values * gain_end, {
            "normalization": True,
            "normalization_peak": peak,
            "normalization_target_gain": target,
            "normalization_gain_start": gain_start,
            "normalization_gain_end": gain_end,
            "normalization_limited": gain_end < 1.0,
        }

    def debug(self) -> dict[str, object]:
        """Return recent node diagnostics as plain copyable data."""

        state = self.transport_signal()
        frontend = self.frontend_position_signal()
        return {
            "node_id": self.id,
            "plot_name": self.plot_node.name,
            "status": state.status,
            "time": state.time,
            "elapsed": state.elapsed,
            "emitted_frames": state.emitted_frames,
            "phase_offset": state.phase_offset,
            "source_epoch": self._source_epoch,
            "sample_rate": state.options.sample_rate,
            "chunk_frames": state.options.chunk_frames,
            "batch_chunks": state.options.batch_chunks,
            "buffer_seconds": state.options.buffer_seconds,
            "last_transition": self._last_transition,
            "phase_match_attempts": self._phase_match_attempts,
            "crossfade_uses": self._crossfade_uses,
            "normalization_gain": self._normalization_state.gain,
            "frontend": asdict(frontend),
            "events": [asdict(event) for event in self._debug_events],
        }

    def clear_debug(self) -> None:
        """Clear accumulated node audio diagnostics."""

        self._debug_events.clear()

    def _record_debug_event(self, kind: str, payload: dict[str, object]) -> None:
        """Append one bounded node diagnostic event."""

        if not self.figure._audio_controller._debug_enabled:
            return
        self._debug_sequence += 1
        self._debug_events.append(
            AudioDebugEvent(
                kind=kind,
                sequence=self._debug_sequence,
                session_id=self.figure._audio_controller.active_output_session_signal(),
                payload=dict(payload),
            )
        )
        del self._debug_events[:-200]


class CurveSound:
    """Expose public playback commands for one curve plot handle."""

    def __init__(self, handle: PlotHandle) -> None:
        """Create a sound facade for a curve plot handle."""

        self._handle = handle
        self._debug = CurveSoundDebug(self)

    def play(
        self,
        *,
        start: float | None = None,
        sample_rate: int = 48_000,
        chunk_frames: int = 2_048,
        batch_chunks: int = 10,
        buffer_seconds: float = 0.25,
        gain: float = 1.0,
        normalization: bool | None = None,
        normalization_ceiling: float = 0.95,
        normalization_attack_seconds: float = 0.005,
        normalization_release_seconds: float = 0.75,
        phase_match: bool = True,
        phase_search_seconds: float = 0.02,
        crossfade_seconds: float = 0.005,
    ) -> CurveSound:
        """Start or restart playback for this curve."""

        node = self._node()
        options = _normalize_options(
            sample_rate=sample_rate,
            chunk_frames=chunk_frames,
            batch_chunks=batch_chunks,
            buffer_seconds=buffer_seconds,
            gain=gain,
            normalization=normalization,
            default_normalization=bool(node.normalization_signal()),
            normalization_ceiling=normalization_ceiling,
            normalization_attack_seconds=normalization_attack_seconds,
            normalization_release_seconds=normalization_release_seconds,
            phase_match=phase_match,
            phase_search_seconds=phase_search_seconds,
            crossfade_seconds=crossfade_seconds,
        )
        if normalization is not None:
            node.set_normalization(options.normalization)
        self._handle.figure._audio_controller.play(node, options, start)
        self._handle.figure._audio_controller.start_frontend_output()
        return self

    @property
    def enabled(self) -> bool:
        """Return whether this curve shows a legend sound control."""

        return bool(self._node().sound_enabled_signal())

    @enabled.setter
    def enabled(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise AudioPlaybackError("Curve sound enabled must be True or False.")
        self._node().sound_enabled_signal.set(value)
        signal = self._handle.figure._sound_control_signal
        signal.set(signal() + 1)

    @property
    def normalization(self) -> bool:
        """Return whether automatic attenuation is enabled for this curve."""

        return bool(self._node().normalization_signal())

    @normalization.setter
    def normalization(self, value: bool) -> None:
        self._node().set_normalization(value)

    def stop(self) -> CurveSound:
        """Stop playback if this curve is the active sound source."""

        controller = self._handle.figure._audio_controller
        if controller.active_node_signal() is self._node():
            controller.stop()
        else:
            self._node().stop()
        return self

    def pause(self) -> CurveSound:
        """Pause this curve if it is selected for playback."""

        controller = self._handle.figure._audio_controller
        node = self._node()
        if controller.active_node_signal() is node:
            controller.pause()
        else:
            node.pause()
        return self

    def resume(self) -> CurveSound:
        """Select this curve and resume playback from the stored time."""

        controller = self._handle.figure._audio_controller
        node = self._node()
        controller.select(node)
        controller.resume()
        return self

    @property
    def time(self) -> float:
        """Return this curve sound's current domain time."""

        return self._node().playback_state().time or 0.0

    @time.setter
    def time(self, value: float) -> None:
        self._node().seek(value)

    def reset(self) -> CurveSound:
        """Seek this curve sound to domain time zero."""

        self._node().reset()
        return self

    def state(self) -> AudioPlaybackState:
        """Return this curve sound's immutable playback state."""

        return self._node().playback_state()

    def _node(self) -> AudioNode:
        """Return the figure-owned audio node for this curve."""

        return self._handle.figure._audio_node_for_plot(self._handle._node)


class CurveSoundDebug:
    """Expose private diagnostics for one curve sound facade."""

    def __init__(self, sound: CurveSound) -> None:
        """Create a private diagnostics namespace for a curve sound."""

        self._sound = sound

    def raw(self) -> dict[str, object]:
        """Return recent raw diagnostics for this curve sound."""

        return self._sound._node().debug()

    def summary(self) -> dict[str, object]:
        """Return a compact troubleshooting summary for this curve."""

        figure_debug = self._sound._handle.figure._audio_controller.debug()
        figure_debug["active_node"] = self._sound._node().debug()
        figure_debug["state"] = asdict(self._sound.state())
        return _audio_debug_summary(figure_debug)

    def clear(self) -> CurveSoundDebug:
        """Clear recent diagnostics for this curve sound."""

        self._sound._node().clear_debug()
        return self

    def enable(self) -> CurveSoundDebug:
        """Enable audio diagnostics for this curve's figure."""

        self._sound._handle.figure._audio_controller.set_debug_enabled(True)
        return self

    def disable(self) -> CurveSoundDebug:
        """Disable audio diagnostics for this curve's figure."""

        self._sound._handle.figure._audio_controller.set_debug_enabled(False)
        return self


class FigureSound:
    """Expose public figure-level audio selection and transport commands."""

    def __init__(self, figure: FigureHandle, controller: FigureAudioController) -> None:
        """Create a sound facade for a figure audio controller."""

        self._figure = figure
        self._controller = controller
        self._debug = FigureSoundDebug(self)

    def stop(self) -> FigureSound:
        """Stop the active figure sound."""

        self._controller.stop()
        return self

    def pause(self) -> FigureSound:
        """Pause the active figure sound."""

        self._controller.pause()
        return self

    def resume(self) -> FigureSound:
        """Resume the selected figure sound."""

        self._controller.resume()
        return self

    @property
    def current(self) -> PlotHandle | None:
        """Return the selected curve plot handle, if one exists."""

        active = self._controller.active_node_signal()
        if active is None:
            return None
        return self._figure._handle_for_node(active.plot_node)

    @current.setter
    def current(self, value: object) -> None:
        node = self._resolve_current(value)
        self._controller.select(self._figure._audio_node_for_plot(node))

    @property
    def time(self) -> float | None:
        """Return the current figure sound's domain time."""

        active = self._controller.active_node_signal()
        if active is None:
            return None
        return active.playback_state().time

    @time.setter
    def time(self, value: float) -> None:
        active = self._controller.active_node_signal()
        if active is not None:
            active.seek(value)

    def reset(self) -> FigureSound:
        """Seek the current figure sound to domain time zero."""

        active = self._controller.active_node_signal()
        if active is not None:
            active.reset()
        return self

    def state(self) -> AudioPlaybackState:
        """Return the figure's immutable audio playback state."""

        return self._controller.state()

    def request_chunk(self, session_id: int, frame_count: int | None = None) -> AudioChunk | None:
        """Return a PCM chunk for a frontend adapter session."""

        return self._controller.request_chunk(session_id, frame_count)

    def _resolve_current(self, value: object) -> PlotNode:
        """Resolve a public current selector to a playable plot node."""

        if isinstance(value, str):
            node = self._figure.plots_by_name.get(value)
            if node is None:
                raise PlotNotFoundForAudioError(
                    f"No plot named {value!r} exists in this figure."
                )
            if node.kind != PLOT_KIND_CURVE:
                raise AudioPlaybackError(
                    f"Plot {value!r} is not playable; only plot(...) curves expose sound."
                )
            return node

        node = getattr(value, "_node", None)
        figure = getattr(value, "figure", None)
        if node is not None:
            if figure is not self._figure:
                raise AudioPlaybackError(
                    "FigureSound.current received a curve from another figure."
                )
            if node.kind != PLOT_KIND_CURVE:
                raise AudioPlaybackError(
                    "FigureSound.current accepts only ordinary plot(...) curve handles."
                )
            return node
        raise AudioPlaybackError(
            "FigureSound.current must be assigned a curve plot handle or plot name."
        )


class FigureSoundDebug:
    """Expose private diagnostics for one figure sound facade."""

    def __init__(self, sound: FigureSound) -> None:
        """Create a private diagnostics namespace for a figure sound."""

        self._sound = sound

    def raw(self) -> dict[str, object]:
        """Return recent raw diagnostics for the figure sound system."""

        return self._sound._controller.debug()

    def summary(self) -> dict[str, object]:
        """Return a compact troubleshooting summary for this figure."""

        return _audio_debug_summary(self.raw())

    def clear(self) -> FigureSoundDebug:
        """Clear recent diagnostics for the figure sound system."""

        self._sound._controller.clear_debug()
        return self

    def enable(self) -> FigureSoundDebug:
        """Enable audio diagnostics for this figure."""

        self._sound._controller.set_debug_enabled(True)
        return self

    def disable(self) -> FigureSoundDebug:
        """Disable audio diagnostics for this figure."""

        self._sound._controller.set_debug_enabled(False)
        return self


class PlotNotFoundForAudioError(AudioPlaybackError):
    """Report a missing named plot during audio selection."""


def _normalize_options(
    *,
    sample_rate: int,
    chunk_frames: int,
    batch_chunks: int,
    buffer_seconds: float,
    gain: float,
    normalization: bool | None,
    default_normalization: bool,
    normalization_ceiling: float,
    normalization_attack_seconds: float,
    normalization_release_seconds: float,
    phase_match: bool,
    phase_search_seconds: float,
    crossfade_seconds: float,
) -> AudioPlaybackOptions:
    """Validate and return playback options."""

    if not isinstance(default_normalization, bool):
        raise PlotSpecError("Curve sound normalization must be True or False.")
    normalized_policy = default_normalization
    if normalization is not None:
        if not isinstance(normalization, bool):
            raise PlotSpecError("Audio normalization must be True or False.")
        normalized_policy = normalization
    if not isinstance(phase_match, bool):
        raise PlotSpecError("phase_match must be True or False.")
    ceiling = _finite_float(normalization_ceiling, "Audio normalization ceiling")
    if ceiling <= 0 or ceiling > 1:
        raise PlotSpecError("Audio normalization ceiling must be between 0 and 1.")
    attack_seconds = _positive_float(
        normalization_attack_seconds,
        "normalization attack seconds",
    )
    release_seconds = _positive_float(
        normalization_release_seconds,
        "normalization release seconds",
    )
    if release_seconds < attack_seconds:
        raise PlotSpecError(
            "Audio normalization release seconds must be greater than or equal "
            "to attack seconds."
        )
    return AudioPlaybackOptions(
        sample_rate=_positive_int(sample_rate, "sample rate"),
        chunk_frames=_positive_int(chunk_frames, "chunk frame count"),
        batch_chunks=_positive_int(batch_chunks, "batch chunk count"),
        buffer_seconds=_positive_float(buffer_seconds, "buffer seconds"),
        gain=_finite_float(gain, "Audio gain"),
        normalization=normalized_policy,
        normalization_ceiling=ceiling,
        normalization_attack_seconds=attack_seconds,
        normalization_release_seconds=release_seconds,
        phase_match=phase_match,
        phase_search_seconds=_nonnegative_float(phase_search_seconds, "phase search seconds"),
        crossfade_seconds=_nonnegative_float(crossfade_seconds, "crossfade seconds"),
    )


def _stopped_transport(start: float) -> AudioTransportState:
    """Return the initial stopped transport state."""

    options = AudioPlaybackOptions()
    return AudioTransportState(
        status="stopped",
        time=start,
        start=start,
        elapsed=0.0,
        emitted_frames=0,
        phase_offset=0.0,
        options=options,
    )


def _tail_from_samples(
    positions: np.ndarray,
    values: np.ndarray,
    sample_rate: int,
) -> AudioTail | None:
    """Keep a short emitted tail for future continuity matching."""

    if positions.size == 0:
        return None
    tail_frames = min(positions.size, max(8, int(0.02 * sample_rate)))
    return AudioTail(
        positions=positions[-tail_frames:].copy(),
        values=values[-tail_frames:].copy(),
    )


def _real_array_like(raw_values: object, shape: tuple[int, ...], *, message: str) -> np.ndarray:
    """Return real float values broadcastable to ``shape``."""

    array = np.asarray(raw_values)
    if np.iscomplexobj(array):
        raise PlotShapeError(message)
    if array.shape == ():
        return np.full(shape, float(array), dtype=float)
    try:
        return np.broadcast_to(array, shape).astype(float)
    except (TypeError, ValueError) as exc:
        raise PlotShapeError(message) from exc


def _positive_int(value: object, name: str) -> int:
    """Return a strictly positive integer option."""

    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PlotSpecError(f"Audio {name} must be a positive integer.")
    return value


def _finite_float(value: object, name: str) -> float:
    """Return a finite float option."""

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PlotSpecError(f"{name} must be a finite real value.") from exc
    if not math.isfinite(result):
        raise PlotSpecError(f"{name} must be a finite real value.")
    return result


def _audio_output_base_class() -> type[object]:
    """Return the installed anywidget base class or a clear fallback."""

    try:
        import anywidget
    except ImportError as exc:
        raise AudioPlaybackError(
            "Audio playback requires anywidget so the browser can request "
            "streamed PCM chunks."
        ) from exc
    return anywidget.AnyWidget


class AudioOutputWidget(_audio_output_base_class()):
    """Bridge figure-owned audio chunks to a browser Web Audio output."""

    _esm = r"""
function pcmFromBuffer(buffer) {
  if (buffer instanceof DataView) {
    return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
  }
  if (buffer instanceof ArrayBuffer) {
    return new Float32Array(buffer);
  }
  if (ArrayBuffer.isView(buffer)) {
    return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
  }
  return new Float32Array(0);
}

function audioOutputStateKey(model) {
  return model?.model_id
    || model?.modelId
    || model?.id
    || model?.get?.("_model_id")
    || null;
}

function audioOutputStateRegistry() {
  if (!globalThis.__mathToolkitAudioOutputStates) {
    globalThis.__mathToolkitAudioOutputStates = new Map();
  }
  return globalThis.__mathToolkitAudioOutputStates;
}

function createAudioOutputState(initialModel, key) {
  let model = initialModel;
  let context = null;
  let workletNode = null;
  let workletReady = false;
  let usingWorklet = false;
  let sessionId = null;
  let sampleRate = 48000;
  let chunkFrames = 2048;
  let batchChunks = 10;
  let bufferSeconds = 0.25;
  let nextStartTime = 0;
  let queuedFrames = 0;
  let timer = null;
  let stopping = false;
  let requestedBatches = 0;
  let sourceNodes = [];
  let debugEnabled = false;
  let disposeTimer = null;
  let handlerModel = null;
  let sourceEpoch = 0;

  function send(type, extra = {}) {
    model.send({ type, session_id: sessionId, ...extra });
  }

  function sendDiagnostic(event, extra = {}) {
    if (sessionId === null || !debugEnabled) {
      return;
    }
    send("diagnostic", {
      event,
      backend: usingWorklet ? "audio-worklet" : "buffer-source",
      queued_seconds: usingWorklet
        ? Math.max(0, queuedFrames / sampleRate)
        : context ? Math.max(0, nextStartTime - context.currentTime) : 0,
      requested_batches: requestedBatches,
      ...extra,
    });
  }

  function stopTimer() {
    if (timer !== null) {
      window.clearInterval(timer);
      timer = null;
    }
  }

  function disposeSession() {
    stopTimer();
    stopping = true;
    requestedBatches = 0;
    sessionId = null;
    nextStartTime = 0;
    queuedFrames = 0;
    sourceEpoch = 0;
    if (workletNode !== null) {
      workletNode.port.postMessage({ type: "reset" });
    }
    for (const source of sourceNodes) {
      try {
        source.stop();
      } catch (_error) {
        // The source may already have ended.
      }
    }
    sourceNodes = [];
  }

  async function ensureWorklet() {
    if (context === null || !context.audioWorklet) {
      return false;
    }
    if (!workletReady) {
      const processor = `
class MathToolkitAudioStreamProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.offset = 0;
    this.queuedFrames = 0;
    this.lowWaterFrames = 4096;
    this.framesUntilRequest = 0;
    this.framesUntilDiagnostic = 0;
    this.active = false;
    this.port.onmessage = (event) => {
      const message = event.data || {};
      if (message.type === "configure") {
        this.lowWaterFrames = Math.max(128, message.low_water_frames || 4096);
        this.active = true;
      } else if (message.type === "chunk") {
        const pcm = message.pcm || new Float32Array(0);
        if (pcm.length > 0) {
          this.queue.push(pcm);
          this.queuedFrames += pcm.length;
        }
      } else if (message.type === "reset" || message.type === "clear") {
        this.queue = [];
        this.offset = 0;
        this.queuedFrames = 0;
        this.framesUntilRequest = 0;
        this.framesUntilDiagnostic = 0;
        if (message.type === "reset") {
          this.active = false;
        }
      }
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0] && outputs[0][0];
    if (!output) {
      return true;
    }
    if (!this.active) {
      output.fill(0);
      return true;
    }

    let written = 0;
    while (written < output.length && this.queue.length > 0) {
      const head = this.queue[0];
      const available = head.length - this.offset;
      const wanted = output.length - written;
      const count = Math.min(available, wanted);
      output.set(head.subarray(this.offset, this.offset + count), written);
      written += count;
      this.offset += count;
      this.queuedFrames -= count;
      if (this.offset >= head.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }
    const missingFrames = output.length - written;
    if (missingFrames > 0) {
      output.fill(0, written);
      this.port.postMessage({
        type: "diagnostic",
        event: "underrun",
        missing_frames: missingFrames,
        queued_frames: this.queuedFrames,
      });
    }

    this.framesUntilRequest -= output.length;
    if (this.queuedFrames < this.lowWaterFrames && this.framesUntilRequest <= 0) {
      this.port.postMessage({ type: "need_data", queued_frames: this.queuedFrames });
      this.framesUntilRequest = Math.max(128, Math.floor(sampleRate * 0.01));
    }
    this.framesUntilDiagnostic -= output.length;
    if (this.framesUntilDiagnostic <= 0) {
      this.port.postMessage({
        type: "diagnostic",
        event: "worklet_queue",
        queued_frames: this.queuedFrames,
      });
      this.framesUntilDiagnostic = Math.max(128, Math.floor(sampleRate * 0.25));
    }
    return true;
  }
}

registerProcessor("math-toolkit-audio-stream", MathToolkitAudioStreamProcessor);
`;
      const blob = new Blob([processor], { type: "application/javascript" });
      const url = URL.createObjectURL(blob);
      try {
        await context.audioWorklet.addModule(url);
        workletReady = true;
      } catch (_error) {
        return false;
      } finally {
        URL.revokeObjectURL(url);
      }
    }
    if (workletNode === null) {
      try {
        workletNode = new AudioWorkletNode(context, "math-toolkit-audio-stream", {
          numberOfInputs: 0,
          numberOfOutputs: 1,
          outputChannelCount: [1],
        });
      } catch (_error) {
        return false;
      }
      workletNode.port.onmessage = (event) => {
        const message = event.data || {};
        if (message.type === "need_data") {
          queuedFrames = message.queued_frames || 0;
          requestMore();
        } else if (message.type === "diagnostic") {
          queuedFrames = message.queued_frames || queuedFrames;
          sendDiagnostic(message.event || "worklet", {
            queued_frames: message.queued_frames || 0,
            missing_frames: message.missing_frames || 0,
          });
        }
      };
      workletNode.connect(context.destination);
    }
    workletNode.port.postMessage({
      type: "configure",
      low_water_frames: Math.max(128, Math.floor(bufferSeconds * sampleRate)),
    });
    return true;
  }

  async function ensureContext() {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) {
      throw new Error("Web Audio is not available in this browser.");
    }
    if (context === null || context.state === "closed") {
      context = new AudioContextClass({ sampleRate });
    }
    if (context.state === "suspended") {
      await context.resume();
    }
    if (context.state !== "running") {
      throw new Error("Web Audio could not enter the running state.");
    }
  }

  function requestMore() {
    if (sessionId === null || stopping) {
      return;
    }
    const batchFrames = Math.max(1, chunkFrames * batchChunks);
    const maxInFlight = Math.min(
      64,
      Math.max(2, Math.ceil((bufferSeconds * sampleRate) / batchFrames) + 2),
    );
    while (requestedBatches < maxInFlight) {
      const queuedSeconds = usingWorklet
        ? Math.max(0, (queuedFrames + requestedBatches * batchFrames) / sampleRate)
        : Math.max(0, nextStartTime - context.currentTime);
      if (queuedSeconds >= bufferSeconds) {
        return;
      }
      requestedBatches += 1;
      sendDiagnostic("batch_request", {
        chunk_frames: chunkFrames,
        chunk_count: batchChunks,
        frame_count: batchFrames,
        queued_seconds_before_request: queuedSeconds,
      });
      send("batch_request", { chunk_count: batchChunks });
    }
  }

  function acceptNewSourceEpoch(nextSourceEpoch) {
    sourceEpoch = nextSourceEpoch;
    sendDiagnostic("source_epoch_accepted", {
      source_epoch: sourceEpoch,
      queued_frames: queuedFrames,
    });
  }

  function scheduleAudioData(message, buffers) {
    if (sessionId === null || message.session_id !== sessionId || stopping) {
      return;
    }
    if (message.type === "batch") {
      requestedBatches = Math.max(0, requestedBatches - 1);
    }
    const chunkSourceEpoch = message.source_epoch || 0;
    if (chunkSourceEpoch < sourceEpoch) {
      sendDiagnostic("audio_data_dropped", {
        reason: "stale_source_epoch",
        chunk_source_epoch: chunkSourceEpoch,
        source_epoch: sourceEpoch,
      });
      requestMore();
      return;
    }
    if (chunkSourceEpoch > sourceEpoch) {
      const availableFrames = message.frames || 0;
      const halfBatchFrames = Math.max(1, Math.floor(chunkFrames * batchChunks / 2));
      if (availableFrames < halfBatchFrames && !message.reached_end) {
        sendDiagnostic("source_switch_deferred", {
          source_epoch: chunkSourceEpoch,
          frames: availableFrames,
          half_batch_frames: halfBatchFrames,
        });
        requestMore();
        return;
      }
      acceptNewSourceEpoch(chunkSourceEpoch);
    }
    const pcm = pcmFromBuffer(buffers && buffers.length ? buffers[0] : null);
    if (pcm.length > 0 && usingWorklet && workletNode !== null) {
      const copy = new Float32Array(pcm);
      queuedFrames += copy.length;
      workletNode.port.postMessage({ type: "chunk", pcm: copy }, [copy.buffer]);
      sendDiagnostic(message.type === "batch" ? "batch_received" : "chunk_received", {
        frames: pcm.length,
        chunk_frames: message.chunk_frames || chunkFrames,
        chunk_count: message.chunk_count || Math.ceil(pcm.length / chunkFrames),
        source_epoch: chunkSourceEpoch,
        reached_end: Boolean(message.reached_end),
      });
    } else if (pcm.length > 0 && context !== null) {
      const audioBuffer = context.createBuffer(1, pcm.length, message.sample_rate || sampleRate);
      audioBuffer.copyToChannel(pcm, 0);
      const source = context.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(context.destination);
      source.onended = () => {
        sourceNodes = sourceNodes.filter((candidate) => candidate !== source);
      };
      const startAt = Math.max(context.currentTime + 0.01, nextStartTime || context.currentTime + 0.01);
      source.start(startAt);
      sourceNodes.push(source);
      nextStartTime = startAt + audioBuffer.duration;
      sendDiagnostic(message.type === "batch" ? "batch_scheduled" : "chunk_scheduled", {
        frames: pcm.length,
        chunk_frames: message.chunk_frames || chunkFrames,
        chunk_count: message.chunk_count || Math.ceil(pcm.length / chunkFrames),
        start_at: startAt,
        source_epoch: chunkSourceEpoch,
        reached_end: Boolean(message.reached_end),
      });
    }
    if (message.reached_end) {
      stopping = true;
      const drainSeconds = usingWorklet
        ? queuedFrames / sampleRate
        : Math.max(0, nextStartTime - context.currentTime);
      window.setTimeout(disposeSession, Math.ceil((drainSeconds + 0.1) * 1000));
      return;
    }
    requestMore();
  }

  async function start(message) {
    disposeSession();
    stopping = false;
    sessionId = message.session_id;
    sampleRate = message.sample_rate || 48000;
    chunkFrames = message.chunk_frames || 2048;
    batchChunks = message.batch_chunks || 10;
    bufferSeconds = message.buffer_seconds || 0.25;
    sourceEpoch = 0;
    debugEnabled = Boolean(message.debug);
    try {
      await ensureContext();
      usingWorklet = await ensureWorklet();
    } catch (error) {
      const name = error && error.name ? error.name : "";
      if (name === "NotAllowedError") {
        send("needs_user_activation", { message: String(error.message || error) });
      } else {
        send("error", { message: String(error.message || error) });
      }
      disposeSession();
      return;
    }
    nextStartTime = context.currentTime + 0.08;
    sendDiagnostic("start", {
      sample_rate: sampleRate,
      chunk_frames: chunkFrames,
      batch_chunks: batchChunks,
      buffer_seconds: bufferSeconds,
    });
    requestMore();
    timer = window.setInterval(() => {
      if (sessionId === null || context === null) {
        return;
      }
      send("position", {
        elapsed: context.currentTime,
        backend: usingWorklet ? "audio-worklet" : "buffer-source",
        queued_seconds: usingWorklet
          ? Math.max(0, queuedFrames / sampleRate)
          : Math.max(0, nextStartTime - context.currentTime),
        requested_batches: requestedBatches,
      });
      requestMore();
    }, 250);
  }

  function handleCustomMessage(message, buffers) {
    if (!message || typeof message !== "object") {
      return;
    }
    if (message.type === "start") {
      start(message);
    } else if (message.type === "chunk" || message.type === "batch") {
      scheduleAudioData(message, buffers || []);
    } else if (message.type === "pause") {
      if (message.session_id === sessionId) {
        disposeSession();
      }
    } else if (message.type === "stale") {
      if (message.session_id === sessionId) {
        disposeSession();
      }
    } else if (message.type === "stop") {
      disposeSession();
    } else if (message.type === "debug") {
      debugEnabled = Boolean(message.enabled);
    }
  }

  function cancelDispose() {
    if (disposeTimer !== null) {
      window.clearTimeout(disposeTimer);
      disposeTimer = null;
    }
  }

  function disposeNow() {
    cancelDispose();
    disposeSession();
    if (context !== null && context.state !== "closed") {
      context.close();
    }
    context = null;
    workletNode = null;
    workletReady = false;
    if (handlerModel !== null && typeof handlerModel.off === "function") {
      handlerModel.off("msg:custom", handleCustomMessage);
    }
    handlerModel = null;
    if (key !== null && audioOutputStateRegistry().get(key) === state) {
      audioOutputStateRegistry().delete(key);
    }
    if (key === null && model.__mathToolkitAudioOutputState === state) {
      model.__mathToolkitAudioOutputState = null;
    }
  }

  function disposeSoon() {
    cancelDispose();
    disposeTimer = window.setTimeout(disposeNow, 5000);
  }

  function setModel(nextModel) {
    if (handlerModel === nextModel) {
      model = nextModel;
      return;
    }
    if (handlerModel !== null && typeof handlerModel.off === "function") {
      handlerModel.off("msg:custom", handleCustomMessage);
    }
    model = nextModel;
    handlerModel = nextModel;
    model.on("msg:custom", handleCustomMessage);
  }

  const state = {
    cancelDispose,
    disposeSoon,
    disposeNow,
    setModel,
  };
  setModel(initialModel);
  return state;
}

function render({ model, el }) {
  el.style.display = "none";

  const key = audioOutputStateKey(model);
  const registry = audioOutputStateRegistry();
  const state = key !== null
    ? registry.get(key) || createAudioOutputState(model, key)
    : model.__mathToolkitAudioOutputState || createAudioOutputState(model, null);
  if (key !== null) {
    registry.set(key, state);
  } else {
    model.__mathToolkitAudioOutputState = state;
  }
  state.setModel(model);
  state.cancelDispose();

  return () => {
    state.disposeSoon();
  };
}

export default { render };
"""

    def __init__(self, figure: FigureHandle) -> None:
        """Create a hidden browser audio bridge for one figure."""

        super().__init__()
        self.figure = figure
        self.on_msg(self._handle_message)

    def send_command(self, content: dict[str, object]) -> None:
        """Send a control command to the browser audio adapter."""

        self.send(content)

    def send_chunk(self, chunk: AudioChunk) -> None:
        """Send one PCM chunk to the browser audio adapter."""

        self.send(
            {
                "type": "chunk",
                "session_id": chunk.session_id,
                "source_epoch": chunk.source_epoch,
                "sample_rate": chunk.sample_rate,
                "frames": int(chunk.pcm.shape[0]),
                "start_time": chunk.start_time,
                "end_time": chunk.end_time,
                "reached_end": chunk.reached_end,
            },
            buffers=[np.ascontiguousarray(chunk.pcm, dtype=np.float32).tobytes()],
        )

    def send_batch(self, batch: AudioBatch) -> None:
        """Send one PCM batch to the browser audio adapter."""

        self.send(
            {
                "type": "batch",
                "session_id": batch.session_id,
                "source_epoch": batch.source_epoch,
                "sample_rate": batch.sample_rate,
                "chunk_frames": batch.chunk_frames,
                "chunk_count": batch.chunk_count,
                "frames": int(batch.pcm.shape[0]),
                "start_time": batch.start_time,
                "end_time": batch.end_time,
                "reached_end": batch.reached_end,
            },
            buffers=[np.ascontiguousarray(batch.pcm, dtype=np.float32).tobytes()],
        )

    def close(self) -> None:
        """Close the widget after stopping browser-side audio."""

        try:
            self.send_command({"type": "stop"})
        finally:
            super().close()

    def _handle_message(
        self,
        widget: object,
        content: dict[str, object],
        buffers: object,
    ) -> None:
        """Forward browser audio requests to the figure controller."""

        self.figure._audio_controller.handle_frontend_message(content, buffers)

def _positive_float(value: object, name: str) -> float:
    """Return a positive finite float option."""

    result = _finite_float(value, f"Audio {name}")
    if result <= 0:
        raise PlotSpecError(f"Audio {name} must be positive.")
    return result


def _nonnegative_float(value: object, name: str) -> float:
    """Return a nonnegative finite float option."""

    result = _finite_float(value, f"Audio {name}")
    if result < 0:
        raise PlotSpecError(f"Audio {name} must be nonnegative.")
    return result


def _semantic_equal(old: object, new: object) -> bool:
    """Return whether two signal values are semantically unchanged."""

    try:
        return bool(old == new)
    except Exception:
        return old is new
