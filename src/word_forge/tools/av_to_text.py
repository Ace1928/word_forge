#!/usr/bin/env python3
"""
High-Performance, Offline Audio/Video Transcription Script

This script converts any audio or video file into a transcript using OpenAI’s Whisper model.
It is designed for local, offline CPU-only usage, while still downloading models as needed.
The transcript is formatted with clear time stamps for each segment, but without overloading
the text with too many time markers.

Dependencies:
    - Python 3.7+
    - openai-whisper (pip install openai-whisper)
    - torch (pip install torch)  [CPU-only usage will be forced]
    - FFmpeg installed and available in your system's PATH

Usage Example:
    python transcribe.py --input path/to/file.mp4 --output transcript.txt --model small
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from datetime import timedelta

try:
    import torch
    import whisper
except ImportError:
    sys.exit(
        "Error: missing required modules. Please install 'openai-whisper' and 'torch'."
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# Supported file extensions for audio and video
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm"}


def is_audio_file(filepath: str) -> bool:
    """Check if file is an audio file based on its extension."""
    _, ext = os.path.splitext(filepath)
    return ext.lower() in AUDIO_EXTENSIONS


def is_video_file(filepath: str) -> bool:
    """Check if file is a video file based on its extension."""
    _, ext = os.path.splitext(filepath)
    return ext.lower() in VIDEO_EXTENSIONS


def extract_audio(input_path: str, output_path: str) -> None:
    """
    Use FFmpeg to extract the audio track from a video file.
    The output audio will be in WAV format with 16kHz sample rate and mono channel.
    """
    logging.info("Extracting audio from video: %s", input_path)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        input_path,
        "-ar",
        "16000",  # set sample rate to 16kHz
        "-ac",
        "1",  # mono channel
        output_path,
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logging.error("FFmpeg error: %s", result.stderr.decode("utf-8"))
        raise RuntimeError("Audio extraction failed.")
    logging.info("Audio extracted to: %s", output_path)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))


def transcribe_audio(audio_path: str, model_size: str) -> dict:
    """
    Load the Whisper model (forcing CPU inference) and transcribe the provided audio file.
    Returns the full transcription result.
    """
    logging.info("Loading Whisper model (%s)...", model_size)
    # Load model; force CPU usage
    device = "cpu"
    model = whisper.load_model(model_size, device=device)

    logging.info("Beginning transcription...")
    # Transcribe with options that preserve segmentation.
    result = model.transcribe(audio_path, fp16=False)  # fp16 must be false for CPU-only
    logging.info("Transcription complete.")
    return result


def generate_formatted_transcript(result: dict) -> str:
    """
    Format the transcription result into a clean transcript.
    Each segment is preceded by a timestamp in HH:MM:SS format.
    """
    transcript_lines = []
    segments = result.get("segments", [])
    if not segments:
        # Fallback to the full text if segments are not available
        transcript_lines.append(result.get("text", ""))
    else:
        for seg in segments:
            start_ts = format_timestamp(seg["start"])
            # Only include a timestamp if the segment is longer than 1 second
            line = f"[{start_ts}] {seg['text'].strip()}"
            transcript_lines.append(line)
    return "\n\n".join(transcript_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Offline transcription of audio/video files using OpenAI's Whisper."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input audio/video file."
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to save the transcript text file."
    )
    parser.add_argument(
        "--model",
        "-m",
        default="large",
        help="Whisper model size (e.g., tiny, base, small, medium, large).",
    )
    args = parser.parse_args()

    input_file = os.path.abspath(args.input)
    if not os.path.exists(input_file):
        logging.error("Input file does not exist: %s", input_file)
        sys.exit(1)

    # Determine if input file is audio or video
    use_temp_audio = False
    if is_audio_file(input_file):
        audio_file = input_file
    elif is_video_file(input_file):
        # Create a temporary file for the extracted audio
        use_temp_audio = True
        tmp_fd, audio_file = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        try:
            extract_audio(input_file, audio_file)
        except Exception as e:
            logging.error("Failed to extract audio: %s", e)
            sys.exit(1)
    else:
        logging.error("Unsupported file type for input: %s", input_file)
        sys.exit(1)

    try:
        # Transcribe audio file
        result = transcribe_audio(audio_file, args.model)
        transcript = generate_formatted_transcript(result)
    except Exception as e:
        logging.error("Transcription failed: %s", e)
        sys.exit(1)
    finally:
        # Clean up temporary audio file if created
        if use_temp_audio and os.path.exists(audio_file):
            os.remove(audio_file)
            logging.info("Temporary audio file removed.")

    # Save transcript to output file
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(transcript)
        logging.info("Transcript saved to: %s", args.output)
    except Exception as e:
        logging.error("Failed to save transcript: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
