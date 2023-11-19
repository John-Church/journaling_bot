import typer
import sounddevice as sd
import numpy as np
import requests
import os
from datetime import datetime
from dotenv import load_dotenv
import soundfile as sf
import threading


load_dotenv()  # This loads the environment variables from .env


app = typer.Typer()


def record_audio(fs=44100):
    typer.echo("🔴 Recording... (press any key to stop)")
    recording = []
    recording_stopped = threading.Event()

    def callback(indata, frames, time, status):
        if recording_stopped.is_set():
            raise sd.CallbackAbort
        recording.append(indata.copy())

    stream = sd.InputStream(samplerate=fs, channels=1, callback=callback)

    with stream:
        stream.start()
        input("Press Enter to stop recording...")  # Wait for keypress
        recording_stopped.set()

    return np.concatenate(recording, axis=0)


from whispercpp import Whisper

# Initialize Whisper with model from .env
whisper_model = os.getenv("WHISPER_MODEL", "medium")
whisper = Whisper(whisper_model)


def speech_to_text(audio_data):
    # Flatten the audio data to one dimension
    flattened_audio = audio_data.flatten()
    sf.write("recording.wav", flattened_audio, 44100)

    try:
        # Transcribe the flattened audio data
        result = whisper.transcribe("recording.wav")
        transcription = whisper.extract_text(result)
        return transcription
    except Exception as e:
        typer.echo(f"Error in Whisper transcription: {e}")
        return ""


def summarize_text(text):
    model = os.getenv("TEXT_MODEL", "mistral")
    data = {
        "model": model,
        "prompt": f"The following text is a recording of the user's thoughts. These thoughts may be somwhat disorganized. Your goal is to organize the thoughts and format them in markdown format. Always quote verbatim when possible and do not add anything that is not present in the original but also you can make some superficial changes to make the sentence flow more natural as long as it does not change the meaning. Feel free to add headers. Here is the text: {text}",
        "stream": False,
    }

    response = requests.post("http://localhost:11434/api/generate", json=data)

    print(response)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        typer.echo("Error in text summarization")
        return ""


@app.command()
def main():
    typer.echo("Welcome!")
    typer.echo("Let us begin.")
    typer.echo("-------------------------------------------------------")

    while True:
        input(
            "\nPress 'Enter' to start recording your thoughts."
        )  # Waits for the user to press Enter

        audio_data = record_audio()  # Starts recording

        typer.echo("Recording stopped. Processing your audio...")

        transcribed_text = speech_to_text(audio_data)
        typer.echo("Transcribed Text: ")
        typer.echo(transcribed_text)

        markdown_text = summarize_text(transcribed_text)
        typer.echo("Markdown Text: ")
        typer.echo(markdown_text)

        if not typer.confirm("Do you want to record again?"):
            journal_path = os.path.expanduser(os.getenv("JOURNAL_PATH", "journal/"))

            if not os.path.exists(journal_path):
                os.makedirs(journal_path)

            file_name = (
                f"{journal_path}Journal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
            with open(file_name, "w") as file:
                file.write(markdown_text)
            typer.echo(f"Saved to {file_name}")
            break


if __name__ == "__main__":
    app()
