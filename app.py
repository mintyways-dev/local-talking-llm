import argparse
import json
import os
import random
import threading
import time
from collections import deque
from queue import Queue
from typing import Any

import numpy as np
import sounddevice as sd
import whisper
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import OllamaLLM
from rich.console import Console

from tts import TextToSpeechService

try:
    import webrtcvad
except ImportError:
    webrtcvad = None

console = Console()
stt = whisper.load_model("base.en")

SAMPLE_RATE = 16000
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".voice_assistant_calibration.json")
SESSION_ID = "voice_assistant_session"


# Parse command line arguments
parser = argparse.ArgumentParser(description="Local Voice Assistant with ChatterBox TTS")
parser.add_argument("--voice", type=str, help="Path to voice sample for cloning")
parser.add_argument("--exaggeration", type=float, default=0.5, help="Emotion exaggeration (0.0-1.0)")
parser.add_argument("--cfg-weight", type=float, default=0.5, help="CFG weight for pacing (0.0-1.0)")
parser.add_argument("--model", type=str, default=None, help="LLM model name (default: gemma3 for ollama, MiniMax-M2.7 for minimax)")
parser.add_argument(
    "--provider",
    type=str,
    default="ollama",
    choices=["ollama", "minimax"],
    help="LLM provider: 'ollama' for local models, 'minimax' for MiniMax cloud API (default: ollama)",
)
parser.add_argument("--api-key", type=str, default=None, help="API key for cloud LLM providers (or set MINIMAX_API_KEY env var)")
parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature (default: 0.7)")
parser.add_argument("--prompt-file", type=str, default="prompt.txt", help="Path to system prompt file (default: prompt.txt)")
parser.add_argument("--save-voice", action="store_true", help="Save generated voice samples")
parser.add_argument("--input-mode", type=str, default="auto", choices=["manual", "auto"], help="Microphone input mode")
parser.add_argument("--vad-mode", type=str, default="hybrid", choices=["energy", "webrtc", "hybrid"], help="Voice activation mode")
parser.add_argument("--calibrate", action="store_true", help="Run microphone calibration and exit")
parser.add_argument("--vad-aggressiveness", type=int, default=2, choices=[0, 1, 2, 3], help="WebRTC VAD aggressiveness")
parser.add_argument("--frame-ms", type=int, default=20, choices=[10, 20, 30], help="Frame size in ms for voice activation")
parser.add_argument("--start-window", type=int, default=3, help="Sliding window size for start trigger")
parser.add_argument("--start-required", type=int, default=2, help="Required active frames in start window")
parser.add_argument("--silence-ms", type=int, default=800, help="Silence duration to end an utterance")
parser.add_argument("--max-utterance-ms", type=int, default=20000, help="Maximum utterance length")
parser.add_argument("--pre-roll-ms", type=int, default=300, help="Audio kept before trigger start")
parser.add_argument("--energy-threshold", type=float, default=None, help="Override calibrated energy threshold")
parser.add_argument("--calibration-path", type=str, default=DEFAULT_CONFIG_PATH, help="Calibration settings file path")
parser.add_argument(
    "--idle-enabled",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable proactive assistant turns during prolonged user silence (auto mode only)",
)
parser.add_argument("--idle-start-seconds", type=float, default=3.0, help="Silence duration before first proactive check")
parser.add_argument("--idle-check-seconds", type=float, default=1.0, help="Interval between proactive checks after idle start")
parser.add_argument("--idle-initial-prob", type=float, default=0.35, help="Initial chance of proactive turn at first idle check")
parser.add_argument("--idle-prob-step", type=float, default=0.10, help="Probability increase after each failed idle check")
parser.add_argument("--idle-prob-max", type=float, default=0.80, help="Maximum proactive probability cap")
parser.add_argument("--idle-max-words", type=int, default=20, help="Maximum words for proactive idle response")
args = parser.parse_args()


# Initialize TTS with ChatterBox
tts = TextToSpeechService()


def create_llm(provider: str, model: str | None = None, api_key: str | None = None, temperature: float = 0.7):
    """
    Create an LLM instance based on the selected provider.

    Args:
        provider: LLM provider name ('ollama' or 'minimax').
        model: Model name. Defaults to 'gemma3' for ollama, 'MiniMax-M2.7' for minimax.
        api_key: API key for cloud providers (or set MINIMAX_API_KEY env var).
        temperature: LLM temperature (default: 0.7).

    Returns:
        A LangChain LLM or ChatModel instance.
    """
    if provider == "ollama":
        return OllamaLLM(model=model or "gemma3", base_url="http://localhost:11434")
    elif provider == "minimax":
        from langchain_openai import ChatOpenAI

        resolved_key = api_key or os.environ.get("MINIMAX_API_KEY")
        if not resolved_key:
            raise ValueError(
                "MiniMax API key is required. Set the MINIMAX_API_KEY environment "
                "variable or pass --api-key on the command line."
            )
        # MiniMax temperature must be in (0.0, 1.0]
        clamped_temperature = max(0.01, min(1.0, temperature))
        return ChatOpenAI(
            model=model or "MiniMax-M2.7",
            base_url="https://api.minimax.io/v1",
            api_key=resolved_key,
            temperature=clamped_temperature,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: ollama, minimax")


# Modern prompt template using ChatPromptTemplate
with open(args.prompt_file, "r", encoding="utf-8") as f:
    prompt_text = f.read()

prompt_template = ChatPromptTemplate.from_messages([
    ("system", prompt_text),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Initialize LLM via provider factory
llm = create_llm(
    provider=args.provider,
    model=args.model,
    api_key=args.api_key,
    temperature=args.temperature,
)

# Create the chain with modern LCEL syntax
# StrOutputParser normalizes output across providers (string from Ollama, AIMessage from ChatOpenAI)
chain = prompt_template | llm | StrOutputParser()

# Chat history storage
chat_sessions: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = InMemoryChatMessageHistory()
    return chat_sessions[session_id]


# Create the runnable with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


def record_audio(stop_event: threading.Event, data_queue: Queue[bytes]) -> None:
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.
    """

    def callback(indata, frames, callback_time, status):  # type: ignore[no-untyped-def]
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        while not stop_event.is_set():
            time.sleep(0.1)


def frame_rms(frame_bytes: bytes) -> float:
    """Compute normalized RMS for a PCM16 mono audio frame."""
    samples = np.frombuffer(frame_bytes, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    float_samples = samples.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(float_samples * float_samples)))


def derive_energy_threshold(
    noise_levels: list[float],
    sigma_multiplier: float = 3.0,
    min_threshold: float = 0.01,
) -> dict[str, float]:
    """Derive an activation threshold from ambient-noise RMS samples."""
    if not noise_levels:
        return {"noise_rms_mean": 0.0, "noise_rms_std": 0.0, "energy_threshold": min_threshold}

    noise_mean = float(np.mean(noise_levels))
    noise_std = float(np.std(noise_levels))
    threshold = max(min_threshold, noise_mean + sigma_multiplier * noise_std)
    return {
        "noise_rms_mean": noise_mean,
        "noise_rms_std": noise_std,
        "energy_threshold": threshold,
    }


def should_start_from_history(recent_active_flags: list[bool], required_active: int) -> bool:
    """Return True when active frames in a sliding window pass trigger requirement."""
    return sum(1 for flag in recent_active_flags if flag) >= required_active


def should_stop_recording(
    silence_frames: int,
    total_frames: int,
    frame_ms: int,
    silence_ms: int,
    max_utterance_ms: int,
) -> bool:
    """Return True when utterance should end on silence or max duration."""
    return silence_frames * frame_ms >= silence_ms or total_frames * frame_ms >= max_utterance_ms


def load_calibration(path: str) -> dict[str, Any]:
    """Load saved calibration settings from disk."""
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_calibration(path: str, config: dict[str, Any]) -> None:
    """Persist calibration settings to disk."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)


def build_vad(vad_mode: str, aggressiveness: int):
    """Create WebRTC VAD instance for webrtc/hybrid modes."""
    if vad_mode in {"webrtc", "hybrid"}:
        if webrtcvad is None:
            raise RuntimeError("webrtcvad is required for webrtc/hybrid modes. Install dependency `webrtcvad`.")
        return webrtcvad.Vad(aggressiveness)
    return None


def frame_is_active(vad_mode: str, vad, frame_bytes: bytes, rms: float, threshold: float) -> bool:
    """Evaluate whether a frame should count as active speech."""
    energy_active = rms >= threshold
    if vad_mode == "energy":
        return energy_active

    is_voiced = bool(vad and vad.is_speech(frame_bytes, SAMPLE_RATE))
    if vad_mode == "webrtc":
        return is_voiced
    return energy_active and is_voiced


def collect_noise_levels(frame_ms: int, duration_sec: float = 3.0) -> list[float]:
    """Capture ambient RMS samples for calibration."""
    frame_samples = int(SAMPLE_RATE * frame_ms / 1000)
    frame_count = max(1, int((duration_sec * 1000) // frame_ms))
    levels: list[float] = []

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        dtype="int16",
        channels=1,
        blocksize=frame_samples,
    ) as stream:
        for _ in range(frame_count):
            frame_bytes, overflowed = stream.read(frame_samples)
            if overflowed:
                console.print("[yellow]Input overflow detected during calibration.[/yellow]")
            levels.append(frame_rms(bytes(frame_bytes)))

    return levels


def validate_calibration(threshold: float, vad_mode: str, vad_aggressiveness: int, frame_ms: int) -> bool:
    """Run a short interactive validation meter for calibration."""
    frame_samples = int(SAMPLE_RATE * frame_ms / 1000)
    frame_count = max(1, int(5000 // frame_ms))
    vad = build_vad(vad_mode, vad_aggressiveness)

    console.print("[cyan]Validation: speak for 5 seconds to test activation.[/cyan]")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        dtype="int16",
        channels=1,
        blocksize=frame_samples,
    ) as stream:
        for idx in range(frame_count):
            frame_bytes, overflowed = stream.read(frame_samples)
            if overflowed:
                console.print("[yellow]Input overflow detected during validation.[/yellow]")
            frame = bytes(frame_bytes)
            rms = frame_rms(frame)
            active = frame_is_active(vad_mode, vad, frame, rms, threshold)
            if idx % max(1, int(1000 // frame_ms)) == 0:
                status = "TRIGGER" if active else "idle"
                console.print(f"[dim]RMS={rms:.4f} threshold={threshold:.4f} [{status}][/dim]")

    confirm = console.input("Accept this calibration? [Y/n/r=retry]: ").strip().lower()
    return confirm not in {"n", "r", "retry"}


def run_calibration(calibration_path: str) -> dict[str, Any]:
    """Perform calibration and persist settings."""
    while True:
        console.print("[cyan]Calibration: stay quiet for 3 seconds...[/cyan]")
        noise_levels = collect_noise_levels(args.frame_ms, duration_sec=3.0)
        metrics = derive_energy_threshold(noise_levels)

        console.print(
            "[green]Noise mean={:.4f}, std={:.4f}, threshold={:.4f}[/green]".format(
                metrics["noise_rms_mean"],
                metrics["noise_rms_std"],
                metrics["energy_threshold"],
            )
        )

        accepted = validate_calibration(
            threshold=metrics["energy_threshold"],
            vad_mode=args.vad_mode,
            vad_aggressiveness=args.vad_aggressiveness,
            frame_ms=args.frame_ms,
        )
        if accepted:
            config: dict[str, Any] = {
                "version": 1,
                "energy_threshold": metrics["energy_threshold"],
                "noise_rms_mean": metrics["noise_rms_mean"],
                "noise_rms_std": metrics["noise_rms_std"],
                "vad_aggressiveness": args.vad_aggressiveness,
                "frame_ms": args.frame_ms,
                "silence_ms": args.silence_ms,
                "start_window": args.start_window,
                "start_required": args.start_required,
                "pre_roll_ms": args.pre_roll_ms,
                "max_utterance_ms": args.max_utterance_ms,
            }
            save_calibration(calibration_path, config)
            console.print(f"[green]Calibration saved to: {calibration_path}[/green]")
            return config

        console.print("[yellow]Retrying calibration...[/yellow]")


def get_activation_config() -> dict[str, Any]:
    """Resolve runtime activation config from defaults, file, and CLI overrides."""
    config = load_calibration(args.calibration_path)

    resolved = {
        "energy_threshold": float(config.get("energy_threshold", 0.02)),
        "vad_aggressiveness": int(config.get("vad_aggressiveness", args.vad_aggressiveness)),
        "frame_ms": int(config.get("frame_ms", args.frame_ms)),
        "silence_ms": int(config.get("silence_ms", args.silence_ms)),
        "start_window": int(config.get("start_window", args.start_window)),
        "start_required": int(config.get("start_required", args.start_required)),
        "pre_roll_ms": int(config.get("pre_roll_ms", args.pre_roll_ms)),
        "max_utterance_ms": int(config.get("max_utterance_ms", args.max_utterance_ms)),
    }

    if args.energy_threshold is not None:
        resolved["energy_threshold"] = float(args.energy_threshold)

    # CLI should always override persisted values for these options.
    resolved["vad_aggressiveness"] = args.vad_aggressiveness
    resolved["frame_ms"] = args.frame_ms
    resolved["silence_ms"] = args.silence_ms
    resolved["start_window"] = args.start_window
    resolved["start_required"] = args.start_required
    resolved["pre_roll_ms"] = args.pre_roll_ms
    resolved["max_utterance_ms"] = args.max_utterance_ms

    resolved["start_window"] = max(1, resolved["start_window"])
    resolved["start_required"] = max(1, min(resolved["start_window"], resolved["start_required"]))
    resolved["pre_roll_ms"] = max(0, resolved["pre_roll_ms"])

    return resolved


def capture_utterance_auto(config: dict[str, Any]) -> np.ndarray:
    """Capture one utterance using voice activation."""
    frame_ms = int(config["frame_ms"])
    frame_samples = int(SAMPLE_RATE * frame_ms / 1000)
    threshold = float(config["energy_threshold"])
    start_window = int(config["start_window"])
    start_required = int(config["start_required"])
    silence_ms = int(config["silence_ms"])
    max_utterance_ms = int(config["max_utterance_ms"])
    pre_roll_frames = int(config["pre_roll_ms"] // frame_ms)

    vad = build_vad(args.vad_mode, int(config["vad_aggressiveness"]))

    trigger_window: deque[bool] = deque(maxlen=start_window)
    pre_roll: deque[bytes] = deque(maxlen=max(1, pre_roll_frames) if pre_roll_frames > 0 else 1)
    captured_frames: list[bytes] = []

    recording = False
    silence_frames = 0
    total_frames = 0

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        dtype="int16",
        channels=1,
        blocksize=frame_samples,
    ) as stream:
        console.print("[cyan]Listening for speech...[/cyan]")

        while True:
            frame_bytes, overflowed = stream.read(frame_samples)
            if overflowed:
                console.print("[yellow]Input overflow detected.[/yellow]")

            frame = bytes(frame_bytes)
            rms = frame_rms(frame)
            is_active = frame_is_active(args.vad_mode, vad, frame, rms, threshold)

            if not recording:
                pre_roll.append(frame)
                trigger_window.append(is_active)
                if len(trigger_window) == start_window and should_start_from_history(list(trigger_window), start_required):
                    recording = True
                    captured_frames.extend(pre_roll)
                    total_frames = len(captured_frames)
                    silence_frames = 0
                    console.print("[green]Speech detected. Recording...[/green]")
                continue

            captured_frames.append(frame)
            total_frames += 1

            if is_active:
                silence_frames = 0
            else:
                silence_frames += 1

            if should_stop_recording(
                silence_frames=silence_frames,
                total_frames=total_frames,
                frame_ms=frame_ms,
                silence_ms=silence_ms,
                max_utterance_ms=max_utterance_ms,
            ):
                break

    audio_data = b"".join(captured_frames)
    return np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0


def capture_utterance_auto_with_timeout(
    config: dict[str, Any],
    listen_timeout_seconds: float | None = None,
) -> np.ndarray:
    """Capture one utterance in auto mode, returning empty audio if no speech starts within timeout."""
    frame_ms = int(config["frame_ms"])
    frame_samples = int(SAMPLE_RATE * frame_ms / 1000)
    threshold = float(config["energy_threshold"])
    start_window = int(config["start_window"])
    start_required = int(config["start_required"])
    silence_ms = int(config["silence_ms"])
    max_utterance_ms = int(config["max_utterance_ms"])
    pre_roll_frames = int(config["pre_roll_ms"] // frame_ms)

    vad = build_vad(args.vad_mode, int(config["vad_aggressiveness"]))

    trigger_window: deque[bool] = deque(maxlen=start_window)
    pre_roll: deque[bytes] = deque(maxlen=max(1, pre_roll_frames) if pre_roll_frames > 0 else 1)
    captured_frames: list[bytes] = []

    recording = False
    silence_frames = 0
    total_frames = 0
    start_wait = time.monotonic()

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        dtype="int16",
        channels=1,
        blocksize=frame_samples,
    ) as stream:
        while True:
            frame_bytes, overflowed = stream.read(frame_samples)
            if overflowed:
                console.print("[yellow]Input overflow detected.[/yellow]")

            frame = bytes(frame_bytes)
            rms = frame_rms(frame)
            is_active = frame_is_active(args.vad_mode, vad, frame, rms, threshold)

            if not recording:
                pre_roll.append(frame)
                trigger_window.append(is_active)
                if len(trigger_window) == start_window and should_start_from_history(list(trigger_window), start_required):
                    recording = True
                    captured_frames.extend(pre_roll)
                    total_frames = len(captured_frames)
                    silence_frames = 0
                    console.print("[green]Speech detected. Recording...[/green]")
                    continue

                if listen_timeout_seconds is not None and listen_timeout_seconds >= 0.0:
                    if time.monotonic() - start_wait >= listen_timeout_seconds:
                        return np.array([], dtype=np.float32)
                continue

            captured_frames.append(frame)
            total_frames += 1

            if is_active:
                silence_frames = 0
            else:
                silence_frames += 1

            if should_stop_recording(
                silence_frames=silence_frames,
                total_frames=total_frames,
                frame_ms=frame_ms,
                silence_ms=silence_ms,
                max_utterance_ms=max_utterance_ms,
            ):
                break

    audio_data = b"".join(captured_frames)
    return np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0


def capture_utterance_manual() -> np.ndarray:
    """Capture one utterance with Enter-to-start/stop behavior."""
    console.input("🎤 Press Enter to start recording, then press Enter again to stop.")

    data_queue: Queue[bytes] = Queue()
    stop_event = threading.Event()
    recording_thread = threading.Thread(
        target=record_audio,
        args=(stop_event, data_queue),
    )
    recording_thread.start()

    input()
    stop_event.set()
    recording_thread.join()

    audio_data = b"".join(list(data_queue.queue))
    return np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.

    Args:
        audio_np (numpy.ndarray): The audio data to be transcribed.

    Returns:
        str: The transcribed text.
    """
    result = stt.transcribe(audio_np, fp16=False)  # Set fp16=True if using a GPU
    text = result["text"].strip()
    return text


def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using the language model.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The generated response.
    """
    # Invoke the chain with history
    response = chain_with_history.invoke(
        {"input": text},
        config={"session_id": SESSION_ID},
    )

    # The response is now a string from the LLM, no need to remove "Assistant:" prefix
    # since we're using a proper chat model setup
    return response.strip()


def _message_to_text(content: Any) -> str:
    """Normalize LangChain message content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(part.strip() for part in parts if part.strip())
    return str(content)


def trim_words(text: str, max_words: int) -> str:
    """Trim text to a maximum number of words."""
    if max_words <= 0:
        return text.strip()
    words = text.strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).strip()


def build_idle_prompt(session_id: str, max_words: int) -> str:
    """Build a direct-model prompt for proactive idle check-ins."""
    history = get_session_history(session_id)
    recent_messages = history.messages[-12:]
    transcript_lines: list[str] = []

    for message in recent_messages:
        role = "User" if message.type == "human" else "Assistant" if message.type == "ai" else message.type.title()
        text = _message_to_text(message.content).strip()
        if text:
            transcript_lines.append(f"{role}: {text}")

    transcript = "\n".join(transcript_lines) if transcript_lines else "(No prior conversation yet)"

    return (
        f"{prompt_text}\n\n"
        "Conversation context:\n"
        f"{transcript}\n\n"
        "The user has been silent for a while. Generate the assistant's next natural message so it continues where you left off or adds to it.\n"
        f"Requirements: one short sentence, at most {max_words} words, context-aware, no filler, no role labels."
    )


def get_idle_llm_response(session_id: str, max_words: int) -> str:
    """Generate a proactive message without injecting a synthetic human turn."""
    prompt = build_idle_prompt(session_id, max_words=max_words)
    raw_response = llm.invoke(prompt)

    if isinstance(raw_response, str):
        text = raw_response.strip()
    else:
        text = _message_to_text(getattr(raw_response, "content", raw_response)).strip()

    text = trim_words(text, max_words=max_words)
    if not text:
        text = "Need anything right now?"

    get_session_history(session_id).add_ai_message(text)
    return text


def init_idle_state(start_seconds: float, initial_probability: float) -> dict[str, float]:
    """Initialize idle check schedule and probability."""
    return {
        "next_check_elapsed_s": max(0.0, start_seconds),
        "current_probability": max(0.0, min(1.0, initial_probability)),
    }


def should_run_idle_check(elapsed_s: float, next_check_elapsed_s: float) -> bool:
    """Return True when elapsed idle time has reached the next check boundary."""
    return elapsed_s >= next_check_elapsed_s


def register_idle_miss(
    idle_state: dict[str, float],
    check_seconds: float,
    probability_step: float,
    probability_max: float,
) -> None:
    """Advance idle schedule/probability after a failed random check."""
    idle_state["next_check_elapsed_s"] += max(0.1, check_seconds)
    idle_state["current_probability"] = min(
        max(0.0, min(1.0, probability_max)),
        idle_state["current_probability"] + max(0.0, probability_step),
    )


def reset_idle_state(idle_state: dict[str, float], start_seconds: float, initial_probability: float) -> None:
    """Reset idle schedule/probability to baseline."""
    idle_state["next_check_elapsed_s"] = max(0.0, start_seconds)
    idle_state["current_probability"] = max(0.0, min(1.0, initial_probability))


def play_audio(sample_rate: int, audio_array: np.ndarray) -> None:
    """
    Plays the given audio data using the sounddevice library.

    Args:
        sample_rate (int): The sample rate of the audio data.
        audio_array (numpy.ndarray): The audio data to be played.

    Returns:
        None
    """
    sd.play(audio_array, sample_rate)
    sd.wait()


def analyze_emotion(text: str) -> float:
    """
    Simple emotion analysis to dynamically adjust exaggeration.
    Returns a value between 0.3 and 0.9 based on text content.
    """
    # Keywords that suggest more emotion
    emotional_keywords = [
        "amazing",
        "terrible",
        "love",
        "hate",
        "excited",
        "sad",
        "happy",
        "angry",
        "wonderful",
        "awful",
        "!",
        "?!",
        "...",
    ]

    emotion_score = 0.5  # Default neutral

    text_lower = text.lower()
    for keyword in emotional_keywords:
        if keyword in text_lower:
            emotion_score += 0.1

    # Cap between 0.3 and 0.9
    return min(0.9, max(0.3, emotion_score))


if __name__ == "__main__":
    console.print("[cyan]🤖 Local Voice Assistant with ChatterBox TTS")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if args.voice:
        console.print(f"[green]Using voice cloning from: {args.voice}")
    else:
        console.print("[yellow]Using default voice (no cloning)")

    console.print(f"[blue]Emotion exaggeration: {args.exaggeration}")
    console.print(f"[blue]CFG weight: {args.cfg_weight}")
    console.print(f"[blue]LLM model: {args.model or ('gemma3' if args.provider == 'ollama' else 'MiniMax-M2.7')}")
    console.print(f"[blue]LLM provider: {args.provider}")
    console.print(f"[blue]Input mode: {args.input_mode}")
    if args.input_mode == "auto":
        console.print(f"[blue]Activation mode: {args.vad_mode}")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("[cyan]Press Ctrl+C to exit.\n")

    # Create voices directory if saving voices
    if args.save_voice:
        os.makedirs("voices", exist_ok=True)

    if args.calibrate:
        run_calibration(args.calibration_path)
        raise SystemExit(0)

    activation_config = get_activation_config()

    if args.input_mode == "auto":
        console.print(
            "[dim]Activation threshold={:.4f}, start={}/{}, silence={}ms, max={}ms[/dim]".format(
                activation_config["energy_threshold"],
                activation_config["start_required"],
                activation_config["start_window"],
                activation_config["silence_ms"],
                activation_config["max_utterance_ms"],
            )
        )

    response_count = 0
    idle_start_seconds = max(0.0, float(args.idle_start_seconds))
    idle_check_seconds = max(0.1, float(args.idle_check_seconds))
    idle_initial_prob = max(0.0, min(1.0, float(args.idle_initial_prob)))
    idle_prob_step = max(0.0, float(args.idle_prob_step))
    idle_prob_max = max(0.0, min(1.0, float(args.idle_prob_max)))
    idle_max_words = max(1, int(args.idle_max_words))
    idle_enabled = bool(args.idle_enabled) and args.input_mode == "auto"
    last_activity_at = time.monotonic()
    idle_state = init_idle_state(idle_start_seconds, idle_initial_prob)

    if idle_enabled:
        console.print(
            "[dim]Idle checks start={}s, cadence={}s, prob={:.0f}% +{:.0f}% (cap {:.0f}%), max words={}[/dim]".format(
                idle_start_seconds,
                idle_check_seconds,
                idle_initial_prob * 100,
                idle_prob_step * 100,
                idle_prob_max * 100,
                idle_max_words,
            )
        )

    try:
        while True:
            if args.input_mode == "manual":
                audio_np = capture_utterance_manual()
            else:
                listen_timeout_seconds = None
                if idle_enabled:
                    elapsed = time.monotonic() - last_activity_at
                    listen_timeout_seconds = max(0.1, idle_state["next_check_elapsed_s"] - elapsed)
                audio_np = capture_utterance_auto_with_timeout(
                    activation_config,
                    listen_timeout_seconds=listen_timeout_seconds,
                )

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="dots"):
                    text = transcribe(audio_np)
                if not text:
                    console.print("[yellow]No speech detected in captured audio.[/yellow]")
                    continue

                console.print(f"[yellow]You: {text}")
                last_activity_at = time.monotonic()
                reset_idle_state(idle_state, idle_start_seconds, idle_initial_prob)

                with console.status("Generating response...", spinner="dots"):
                    response = get_llm_response(text)

                    # Analyze emotion and adjust exaggeration dynamically
                    dynamic_exaggeration = analyze_emotion(response)

                    # Use lower cfg_weight for more expressive responses
                    dynamic_cfg = args.cfg_weight * 0.8 if dynamic_exaggeration > 0.6 else args.cfg_weight

                    sample_rate, audio_array = tts.long_form_synthesize(
                        response,
                        audio_prompt_path=args.voice,
                        exaggeration=dynamic_exaggeration,
                        cfg_weight=dynamic_cfg,
                    )

                console.print(f"[cyan]Assistant: {response}")
                console.print(f"[dim](Emotion: {dynamic_exaggeration:.2f}, CFG: {dynamic_cfg:.2f})[/dim]")

                # Save voice sample if requested
                if args.save_voice:
                    response_count += 1
                    filename = f"voices/response_{response_count:03d}.wav"
                    tts.save_voice_sample(response, filename, args.voice)
                    console.print(f"[dim]Voice saved to: {filename}[/dim]")

                play_audio(sample_rate, audio_array)
                last_activity_at = time.monotonic()
                reset_idle_state(idle_state, idle_start_seconds, idle_initial_prob)
            else:
                if not idle_enabled:
                    console.print("[red]No audio recorded. Please ensure your microphone is working.[/red]")
                    continue

                elapsed = time.monotonic() - last_activity_at
                if not should_run_idle_check(elapsed, idle_state["next_check_elapsed_s"]):
                    continue

                chance = idle_state["current_probability"]
                roll = random.random()
                if roll >= chance:
                    register_idle_miss(idle_state, idle_check_seconds, idle_prob_step, idle_prob_max)
                    console.print(
                        "[dim]Idle check skipped ({:.0f}% chance, roll {:.2f}). Next check at {:.1f}s.[/dim]".format(
                            chance * 100,
                            roll,
                            idle_state["next_check_elapsed_s"],
                        )
                    )
                    continue

                with console.status("Generating proactive response...", spinner="dots"):
                    response = get_idle_llm_response(SESSION_ID, max_words=idle_max_words)
                    dynamic_exaggeration = analyze_emotion(response)
                    dynamic_cfg = args.cfg_weight * 0.8 if dynamic_exaggeration > 0.6 else args.cfg_weight
                    sample_rate, audio_array = tts.long_form_synthesize(
                        response,
                        audio_prompt_path=args.voice,
                        exaggeration=dynamic_exaggeration,
                        cfg_weight=dynamic_cfg,
                    )

                console.print(f"[cyan]Assistant: {response}")
                console.print(
                    "[dim](Proactive idle turn | chance: {:.0f}%, Emotion: {:.2f}, CFG: {:.2f})[/dim]".format(
                        chance * 100,
                        dynamic_exaggeration,
                        dynamic_cfg,
                    )
                )

                if args.save_voice:
                    response_count += 1
                    filename = f"voices/response_{response_count:03d}.wav"
                    tts.save_voice_sample(response, filename, args.voice)
                    console.print(f"[dim]Voice saved to: {filename}[/dim]")

                play_audio(sample_rate, audio_array)
                last_activity_at = time.monotonic()
                reset_idle_state(idle_state, idle_start_seconds, idle_initial_prob)

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...[/red]")

    console.print("[blue]Session ended. Thank you for using ChatterBox Voice Assistant![/blue]")
