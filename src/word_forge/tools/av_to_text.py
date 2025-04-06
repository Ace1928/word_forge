#!/usr/bin/env python3
"""
High-Performance, Offline Audio/Video Transcription Script

This script converts any audio or video file into a transcript using OpenAI's Whisper model.
It is designed for local, offline CPU-only usage, while still downloading models as needed.
The transcript is formatted with clear time stamps for each segment, but without overloading
the text with too many time markers.

Dependencies:
    - Python 3.7+
    - openai-whisper (pip install openai-whisper)
    - torch (pip install torch)  [CPU-only usage will be forced]
    - FFmpeg installed and available in your system's PATH

Usage Example:
    python av_to_txt.py --input path/to/file.mp4 --output path/to/{file}.txt --model large
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from datetime import timedelta
from typing import List, Set, Tuple, cast

# Conditionally import Whisper to handle potential import failures gracefully
import whisper  # type: ignore

# For Python 3.7-3.8 compatibility
try:
    from typing import Literal, TypedDict  # Python 3.8+
except ImportError:
    from typing_extensions import Literal, TypedDict  # Python 3.7


class TranscriptionSegment(TypedDict):
    """A segment from a Whisper transcription result."""

    text: str
    start: float
    end: float


class TranscriptionResult(TypedDict):
    """The complete result from a Whisper transcription."""

    text: str
    segments: List[TranscriptionSegment]


class TranscriptionError(Exception):
    """Exception raised for errors during the transcription process."""

    pass


class AudioExtractionError(Exception):
    """Exception raised when audio extraction from video fails."""

    pass


class FileOperationError(Exception):
    """Exception raised for file-related operations."""

    pass


class ModelSizeError(ValueError):
    """Exception raised when an invalid model size is specified."""

    pass


# Define valid model size type for stricter type checking
ModelSize = Literal["tiny", "base", "small", "medium", "large"]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# Supported file extensions for audio and video
AUDIO_EXTENSIONS: Set[str] = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
VIDEO_EXTENSIONS: Set[str] = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm"}

# Valid Whisper model sizes
VALID_MODEL_SIZES: Set[str] = {"tiny", "base", "small", "medium", "large"}


def is_audio_file(filepath: str) -> bool:
    """
    Check if file is an audio file based on its extension.

    Args:
        filepath: Path to the file to check

    Returns:
        True if the file has a recognized audio extension, False otherwise
    """
    _, ext = os.path.splitext(filepath)
    return ext.lower() in AUDIO_EXTENSIONS


def is_video_file(filepath: str) -> bool:
    """
    Check if file is a video file based on its extension.

    Args:
        filepath: Path to the file to check

    Returns:
        True if the file has a recognized video extension, False otherwise
    """
    _, ext = os.path.splitext(filepath)
    return ext.lower() in VIDEO_EXTENSIONS


def validate_model_size(model_size: str) -> str:
    """
    Validate that the provided model size is supported by Whisper.

    Args:
        model_size: The model size to validate

    Returns:
        The validated model size (unchanged)

    Raises:
        ModelSizeError: If the model size is not valid
    """
    if model_size.lower() not in VALID_MODEL_SIZES:
        valid_sizes = ", ".join(sorted(VALID_MODEL_SIZES))
        raise ModelSizeError(
            f"Invalid model size: '{model_size}'. Valid options are: {valid_sizes}"
        )
    return model_size


def extract_audio(input_path: str, output_path: str) -> None:
    """
    Use FFmpeg to extract the audio track from a video file.

    The output audio will be in WAV format with 16kHz sample rate and mono channel,
    which is optimized for the Whisper model.

    Args:
        input_path: Path to the input video file
        output_path: Path where the extracted audio will be saved

    Raises:
        AudioExtractionError: If the audio extraction process fails
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

    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )

    if result.returncode != 0:
        error_message = result.stderr.decode("utf-8").strip()
        logging.error("FFmpeg error: %s", error_message)
        raise AudioExtractionError(
            f"Audio extraction failed: {error_message or 'unknown FFmpeg error'}"
        )

    logging.info("Audio extracted to: %s", output_path)


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS format.

    Args:
        seconds: Time in seconds to be formatted

    Returns:
        Formatted time string in the format HH:MM:SS

    Examples:
        >>> format_timestamp(3661.5)
        '1:01:01'
        >>> format_timestamp(70.2)
        '0:01:10'
    """
    return str(timedelta(seconds=int(seconds)))


def transcribe_audio(audio_path: str, model_size: str) -> TranscriptionResult:
    """
    Load the Whisper model and transcribe the provided audio file.

    This function forces CPU inference for compatibility across systems without
    dedicated GPUs.

    Args:
        audio_path: Path to the audio file to transcribe
        model_size: Size of the Whisper model to use (tiny, base, small, medium, large)

    Returns:
        Dictionary containing the transcription result with segments and text

    Raises:
        TranscriptionError: If the transcription process fails
        ModelSizeError: If the model size is not valid
    """
    # Validate model size
    validate_model_size(model_size)

    logging.info("Loading Whisper model (%s)...", model_size)

    try:
        # Load model; force CPU usage for compatibility
        device = "cpu"
        model = whisper.load_model(model_size, device=device)

        logging.info("Beginning transcription...")
        # Transcribe with options that preserve segmentation.
        # fp16 must be false for CPU-only inference
        result = model.transcribe(audio_path, fp16=False)  # type: ignore
        logging.info("Transcription complete.")

        # Cast to TranscriptionResult to ensure type safety
        return cast(TranscriptionResult, result)
    except Exception as e:
        logging.error("Transcription failed: %s", str(e))
        raise TranscriptionError(f"Failed to transcribe audio: {str(e)}")


def generate_formatted_transcript(result: TranscriptionResult) -> str:
    """
    Format the transcription result into a clean transcript.

    Each segment is preceded by a timestamp in HH:MM:SS format.

    Args:
        result: The transcription result dictionary from Whisper

    Returns:
        Formatted transcript text with timestamps
    """
    transcript_lines: List[str] = []
    segments = result.get("segments", [])

    if not segments:
        # Fallback to the full text if segments are not available
        full_text = result.get("text", "")
        if full_text:
            transcript_lines.append(full_text)
        else:
            logging.warning("No segments or full text found in transcription result")
    else:
        for seg in segments:
            start_ts = format_timestamp(seg["start"])
            line = f"[{start_ts}] {seg['text'].strip()}"
            transcript_lines.append(line)

    return "\n\n".join(transcript_lines)


def prepare_audio_file(input_file: str) -> Tuple[str, bool]:
    """
    Prepare the audio file for transcription, extracting from video if needed.

    Args:
        input_file: Path to the input file (audio or video)

    Returns:
        Tuple containing:
            - Path to the audio file to be transcribed
            - Boolean indicating if a temporary file was created

    Raises:
        FileOperationError: If the file type is unsupported or processing fails
    """
    if is_audio_file(input_file):
        return input_file, False

    if is_video_file(input_file):
        # Create a temporary file for the extracted audio
        tmp_fd, audio_file = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)

        try:
            extract_audio(input_file, audio_file)
            return audio_file, True
        except AudioExtractionError as e:
            # Clean up the temporary file if extraction failed
            if os.path.exists(audio_file):
                os.remove(audio_file)
            raise FileOperationError(f"Failed to extract audio: {str(e)}")

    raise FileOperationError(
        f"Unsupported file type for input: {input_file}. "
        f"Supported audio: {', '.join(sorted(AUDIO_EXTENSIONS))}, "
        f"Supported video: {', '.join(sorted(VIDEO_EXTENSIONS))}"
    )


def save_transcript(transcript: str, output_path: str, encoding: str = "utf-8") -> None:
    """
    Save the transcript to a text file.

    Args:
        transcript: The formatted transcript text
        output_path: Path where the transcript should be saved
        encoding: Character encoding to use when writing the file (default: utf-8)

    Raises:
        FileOperationError: If saving the transcript fails
    """
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, "w", encoding=encoding) as f:
            f.write(transcript)
        logging.info("Transcript saved to: %s", output_path)
    except OSError as e:
        raise FileOperationError(f"Failed to save transcript: {e.strerror}")
    except Exception as e:
        raise FileOperationError(f"Failed to save transcript: {str(e)}")


def process_file(input_file: str, output_file: str, model_size: str) -> None:
    """
    Process an audio/video file to generate a transcript.

    This function handles the complete pipeline: preparing the audio,
    transcribing it, and saving the transcript.

    Args:
        input_file: Path to the input audio or video file
        output_file: Path where the transcript should be saved
        model_size: Whisper model size to use

    Raises:
        FileNotFoundError: If the input file doesn't exist
        FileOperationError: If file operations fail
        TranscriptionError: If transcription fails
        ModelSizeError: If the model size is invalid
    """
    # Validate input file existence
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    if not os.path.isfile(input_file):
        raise FileOperationError(f"Input path is not a file: {input_file}")

    audio_file = None
    use_temp_audio = False

    try:
        # Prepare audio file (extract from video if needed)
        audio_file, use_temp_audio = prepare_audio_file(input_file)

        # Transcribe audio
        result = transcribe_audio(audio_file, model_size)
        transcript = generate_formatted_transcript(result)

        # Save transcript to output file
        save_transcript(transcript, output_file)

    finally:
        # Clean up temporary audio file if created
        if use_temp_audio and audio_file and os.path.exists(audio_file):
            try:
                os.remove(audio_file)
                logging.info("Temporary audio file removed.")
            except OSError as e:
                logging.warning("Failed to remove temporary file: %s", str(e))


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    When no arguments are provided, the script will run in interactive mode.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Offline transcription of audio/video files using OpenAI's Whisper."
    )
    parser.add_argument(
        "--input", "-i", help="Path to input audio/video file or directory."
    )
    parser.add_argument(
        "--output", "-o", help="Path to save the transcript text file or directory."
    )
    parser.add_argument(
        "--model",
        "-m",
        default="tiny",
        choices=sorted(VALID_MODEL_SIZES),
        help="Whisper model size (e.g., tiny, base, small, medium, large).",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Process directories recursively (when input is a directory).",
    )
    return parser.parse_args()


def interactive_mode() -> Tuple[str, str, str, bool]:
    """
    Provide interactive interface for file selection and transcription options.

    Guides users through selecting input files/directories, output locations,
    model size, and recursive processing options.

    Returns:
        Tuple[str, str, str, bool]: Input path, output path, model size, and recursive flag

    Raises:
        SystemExit: If user cancels the operation
    """
    print("\n=== Audio/Video Transcription Tool ===\n")

    # Input selection
    print("Select input file or directory:")
    input_path = input("Path: ").strip()
    while not os.path.exists(input_path):
        print("Error: Path does not exist.")
        input_path = input("Path: ").strip()
    input_path = os.path.abspath(input_path)

    # Recursive option for directories
    recursive = False
    if os.path.isdir(input_path):
        recursive_choice = (
            input("Process directory recursively? (y/n): ").strip().lower()
        )
        recursive = recursive_choice.startswith("y")

    # Output selection
    is_dir = os.path.isdir(input_path)
    prompt = "Select output directory:" if is_dir else "Select output file path:"
    print(f"\n{prompt}")
    output_path = input("Path: ").strip()

    # Model selection
    print("\nSelect Whisper model size:")
    model_sizes = sorted(VALID_MODEL_SIZES)
    for i, size in enumerate(model_sizes, 1):
        print(f"{i}. {size}")

    model_choice = 0
    while model_choice < 1 or model_choice > len(model_sizes):
        try:
            model_choice = int(input("Enter number (1-5): "))
        except ValueError:
            model_choice = 0

    model_size = model_sizes[model_choice - 1]

    # Confirmation
    print("\nConfiguration:")
    print(f"• Input: {input_path}")
    print(f"• Output: {output_path}")
    print(f"• Model: {model_size}")
    if os.path.isdir(input_path):
        print(f"• Recursive: {'Yes' if recursive else 'No'}")

    confirm = input("\nProceed with transcription? (y/n): ").strip().lower()
    if not confirm.startswith("y"):
        print("Operation cancelled.")
        sys.exit(0)

    return input_path, output_path, model_size, recursive


def process_directory(
    input_dir: str, output_dir: str, model_size: str, recursive: bool = False
) -> None:
    """
    Process all compatible audio/video files in a directory.

    Args:
        input_dir: Path to directory containing media files
        output_dir: Output directory for transcripts
        model_size: Whisper model size to use
        recursive: Whether to process subdirectories

    Raises:
        FileOperationError: If directory operations fail
    """
    if not os.path.exists(input_dir):
        raise FileOperationError(f"Input directory does not exist: {input_dir}")

    if not os.path.isdir(input_dir):
        raise FileOperationError(f"Input path is not a directory: {input_dir}")

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Collect media files
    media_files: List[str] = []

    if recursive:
        for root, _, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if is_audio_file(file_path) or is_video_file(file_path):
                    media_files.append(file_path)
    else:
        for item in os.listdir(input_dir):
            file_path = os.path.join(input_dir, item)
            if os.path.isfile(file_path) and (
                is_audio_file(file_path) or is_video_file(file_path)
            ):
                media_files.append(file_path)

    if not media_files:
        logging.warning(f"No supported media files found in {input_dir}")
        return

    total_files = len(media_files)
    logging.info(f"Found {total_files} media files to process")

    # Process each file
    for i, media_file in enumerate(media_files, 1):
        try:
            # Generate output path preserving directory structure
            rel_path = os.path.relpath(media_file, input_dir)
            base_name = os.path.splitext(os.path.basename(media_file))[0]

            if os.path.dirname(rel_path) and recursive:
                sub_dir = os.path.dirname(rel_path)
                output_subdir = os.path.join(output_dir, sub_dir)
                os.makedirs(output_subdir, exist_ok=True)
                output_file = os.path.join(output_subdir, f"{base_name}.txt")
            else:
                output_file = os.path.join(output_dir, f"{base_name}.txt")

            logging.info(f"Processing file {i}/{total_files}: {media_file}")
            process_file(media_file, output_file, model_size)

        except Exception as e:
            logging.error(f"Error processing {media_file}: {str(e)}")


def main() -> int:
    """
    Main entry point for the transcription script.

    Processes command-line arguments or runs in interactive mode.
    Handles both individual files and directories with appropriate error handling.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        args = parse_arguments()

        # Check if we should run in interactive mode
        if args.input is None or args.output is None:
            input_path, output_path, model_size, recursive = interactive_mode()
        else:
            input_path = os.path.abspath(args.input)
            output_path = args.output
            model_size = args.model
            recursive = args.recursive

        # Process file or directory based on input type
        if os.path.isdir(input_path):
            # Ensure output is a directory when input is a directory
            if output_path.endswith((".txt", ".md")):
                raise FileOperationError(
                    "Output must be a directory when input is a directory"
                )
            process_directory(input_path, output_path, model_size, recursive)
        else:
            process_file(input_path, output_path, model_size)

        logging.info("Transcription completed successfully")
        return 0

    except FileNotFoundError as e:
        logging.error(str(e))
        return 1
    except FileOperationError as e:
        logging.error(str(e))
        return 1
    except TranscriptionError as e:
        logging.error(str(e))
        return 1
    except AudioExtractionError as e:
        logging.error(str(e))
        return 1
    except ModelSizeError as e:
        logging.error(str(e))
        return 1
    except KeyboardInterrupt:
        logging.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
