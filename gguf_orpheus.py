import os
import sys
import requests
import json
import time
import wave
import numpy as np
import sounddevice as sd
import argparse
import threading
import queue
import asyncio
from flask import Flask, request, jsonify, stream_with_context, Response
import re  # Import the regular expression module

# LM Studio API settings (These will now be configurable via arguments, defaults are kept)
DEFAULT_API_URL_PREFIX = "http://127.0.0.1:1234"
API_COMPLETIONS_ENDPOINT = "/v1/completions"
HEADERS = {
    "Content-Type": "application/json"
}

# Model parameters (Keep these as they are, but model name will be configurable)
DEFAULT_MODEL_NAME = "orpheus-3b-0.1-ft-q4_k_m"
MAX_TOKENS = 1200
TEMPERATURE = 0.6
TOP_P = 0.9
REPETITION_PENALTY = 1.1
SAMPLE_RATE = 24000  # SNAC model uses 24kHz

# Available voices (Keep these as they are)
AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
DEFAULT_VOICE = "tara"  # Best voice according to documentation

# Special token IDs (Keep these as they are)
START_TOKEN_ID = 128259
END_TOKEN_IDS = [128009, 128260, 128261, 128257]
CUSTOM_TOKEN_PREFIX = "<custom_token_"

app = Flask(__name__)

def format_prompt(prompt, voice=DEFAULT_VOICE):
    """Format prompt for Orpheus model with voice prefix and special tokens."""
    if voice not in AVAILABLE_VOICES:
        print(f"Warning: Voice '{voice}' not recognized. Using '{DEFAULT_VOICE}' instead.")
        voice = DEFAULT_VOICE

    # Format similar to how engine_class.py does it with special tokens
    formatted_prompt = f"{voice}: {prompt}"

    # Add special token markers for the LM Studio API (Keep these as they are)
    special_start = "<|audio|>"  # Using the additional_special_token from config
    special_end = "<|eot_id|>"   # Using the eos_token from config

    return f"{special_start}{formatted_prompt}{special_end}"

def generate_tokens_from_api(prompt, api_url_prefix, model_name, voice=DEFAULT_VOICE, temperature=TEMPERATURE,
                            top_p=TOP_P, max_tokens=MAX_TOKENS, repetition_penalty=REPETITION_PENALTY):
    """Generate tokens from text using LM Studio API."""
    formatted_prompt = format_prompt(prompt, voice)
    print(f"Generating speech for: {formatted_prompt}")

    api_url = api_url_prefix + API_COMPLETIONS_ENDPOINT # Construct the full API URL

    # Create the request payload for the LM Studio API (Keep these as they are)
    payload = {
        "model": model_name, # Use the model name from arguments
        "prompt": formatted_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repetition_penalty,
        "stream": True
    }

    # Make the API request with streaming (Keep these as they are)
    response = requests.post(api_url, headers=HEADERS, json=payload, stream=True, timeout=60)

    if response.status_code != 200:
        print(f"Error: API request failed with status code {response.status_code}")
        print(f"Error details: {response.text}")
        return

    # Process the streamed response (Keep these as they are)
    token_counter = 0
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data_str = line[6:]  # Remove the 'data: ' prefix
                if data_str.strip() == '[DONE]':
                    break

                try:
                    data = json.loads(data_str)
                    if 'choices' in data and len(data['choices']) > 0:
                        token_text = data['choices'][0].get('text', '')
                        token_counter += 1
                        if token_text:
                            yield token_text
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue

    print("Token generation complete")

def turn_token_into_id(token_string, index):
    """Convert token string to numeric ID for audio processing."""
    # Strip whitespace
    token_string = token_string.strip()

    # Find the last token in the string
    last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)

    if last_token_start == -1:
        return None

    # Extract the last token
    last_token = token_string[last_token_start:]

    # Process the last token
    if last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            token_id = int(number_str) - 10 - ((index % 7) * 4096)
            return token_id
        except ValueError:
            return None
    else:
        return None

def convert_to_audio(multiframe, count):
    """Convert token frames to audio."""
    from decoder import convert_to_audio as orpheus_convert_to_audio # keep import here
    return orpheus_convert_to_audio(multiframe, count)


def tokens_decoder_sync_generator(syn_token_gen):
    """Synchronous token decoder that converts token stream to audio byte stream.
       Modified to yield ALL audio bytes as ONE single chunk at the end, instead of streaming segments.
    """
    audio_segments = [] # To collect all audio segments for saving to file
    buffer = []
    count = 0
    for token_text in syn_token_gen:
        token = turn_token_into_id(token_text, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1

            # Convert to audio when we have enough tokens
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                audio_samples = convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    audio_segments.append(audio_samples) # Append segment for file saving

    # After generator finishes, yield ALL audio bytes as ONE single chunk
    print("Tokens decoded, yielding ALL audio bytes as single chunk.") # Added log
    yield b''.join(audio_segments) # Yield all segments joined together at the end


def generate_speech_from_api_generator(prompt, api_url_prefix, model_name, voice=DEFAULT_VOICE, temperature=TEMPERATURE,
                     top_p=TOP_P, max_tokens=MAX_TOKENS, repetition_penalty=REPETITION_PENALTY):
    """Generate speech from text using Orpheus model via LM Studio API and return audio byte stream generator."""
    token_generator = generate_tokens_from_api(
        prompt=prompt,
        api_url_prefix=api_url_prefix, # Pass API URL prefix
        model_name=model_name,       # Pass model name
        voice=voice,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty
    )
    return tokens_decoder_sync_generator(token_generator)


@app.route('/v1/audio/speech', methods=['POST'])  # Explicitly define the endpoint as '/v1/audio/speech'
def speech_endpoint():
    """
    Endpoint for speech generation, matching OpenWebUI's expected URL:
    /v1/audio/speech
    Expects text in the 'input' field of the JSON request (OpenWebUI format).
    Modified to: 1) Save audio to disk FIRST. 2) Then send the SAVED file as response.
    """
    print("Request received at /v1/audio/speech")  # Log request arrival
    try:
        content_type = request.headers.get('Content-Type')
        print(f"Content-Type: {content_type}") # Log Content-Type
        if content_type != 'application/json':
            print(f"Error: Expected application/json, but got: {content_type}")
            return jsonify({"error": "Expected Content-Type: application/json"}), 400

        data = request.get_json()
        print(f"Received JSON data: {data}") # Log received JSON data

        text_input = data.get('input') # Try to get text from 'input' field (OpenWebUI)
        if not text_input:
            text_input = data.get('text') # Fallback to 'text' field (if you might use it elsewhere)

        if not text_input:
            print("Error: Missing 'input' or 'text' in request body") # Log missing text error
            return jsonify({"error": "Missing 'input' or 'text' in request body"}), 400

        # --- Remove extra spaces from input text ---
        text_input = text_input.strip() # Remove leading/trailing spaces
        text_input = " ".join(text_input.split()) # Normalize internal spaces (optional, but good practice)

        print(f"Extracted text from input: {text_input}") # Log extracted text

        voice = data.get('voice', DEFAULT_VOICE) # Get voice from request, default to DEFAULT_VOICE
        temperature = data.get('temperature', TEMPERATURE)
        top_p = data.get('top_p', TOP_P)
        repetition_penalty = data.get('repetition_penalty', REPETITION_PENALTY)
        max_tokens = data.get('max_tokens', MAX_TOKENS)

        # --- Get API URL prefix and model name from app config, fallback to defaults if not set ---
        api_url_prefix = app.config.get('API_URL_PREFIX', DEFAULT_API_URL_PREFIX)
        model_name = app.config.get('MODEL_NAME', DEFAULT_MODEL_NAME)

        # --- Filename and Output Folder Logic ---
        output_folder = "output"
        os.makedirs(output_folder, exist_ok=True) # Create 'output' folder if it doesn't exist

        # Get first two words for filename
        first_two_words = " ".join(text_input.strip().split()[:2])
        safe_filename = re.sub(r'[^a-zA-Z0-9_]', '_', first_two_words) # Sanitize filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file_path = os.path.join(output_folder, f"{safe_filename}_{timestamp}.wav")
        print(f"Saving audio to: {output_file_path}") # Log saving path

        def generate(): # Modified generate function - now just saves to file and returns filepath
            all_audio_bytes = b'' # To accumulate all audio bytes
            audio_generator = generate_speech_from_api_generator(
                prompt=text_input, # Use the extracted text_input here
                api_url_prefix=api_url_prefix, # Use configured API URL prefix
                model_name=model_name,       # Use configured model name
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens
            )
            for audio_chunk in audio_generator: # Expecting only ONE chunk now from tokens_decoder_sync_generator
                if isinstance(audio_chunk, bytes):
                    all_audio_bytes += audio_chunk # Accumulate
                else:
                    print(f"Warning: Unexpected type from audio_generator: {type(audio_chunk)}")

            print("Generate function finished, SAVING audio to file.") # Log saving

            # --- Save WAV file ---
            if all_audio_bytes: # Only save if there's audio data
                with wave.open(output_file_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2) # Assuming 16-bit PCM
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(all_audio_bytes)
                print(f"Audio saved to: {output_file_path}")
            else:
                print("No audio data generated, skipping file save.")
                return None # Indicate no file path to send

            return output_file_path # Return the filepath of the saved file


        # --- Generate and get output filepath ---
        saved_file_path = generate() # Now generate() returns filepath

        if saved_file_path: # Check if filepath was returned (meaning audio was generated and saved)
            print(f"Reading saved audio file: {saved_file_path} for response.") # Log file reading
            with open(saved_file_path, 'rb') as audio_file: # Open in binary read mode
                audio_response_data = audio_file.read() # Read all file content into bytes
            print("Sending saved audio file content as response.") # Log response sending
            return Response(audio_response_data, mimetype='audio/wav') # Send file content as response
        else:
            return jsonify({"error": "Audio generation failed, no audio file saved."}), 500 # Error if no file


    except Exception as e:
        error_message = f"Exception in speech_endpoint: {e}"
        print(error_message) # Log any exceptions
        return jsonify({"error": error_message}), 500 # Return 500 for internal server error


@app.route('/v1/audio/voices', methods=['GET']) # Changed to /v1/audio/voices
def voices_endpoint():
    """Returns a list of available voices for /v1/audio/voices endpoint"""
    available_voices_list = [{"name": voice, "id": voice} for voice in AVAILABLE_VOICES] #openai style voices list
    print(f"Serving voices list: {available_voices_list} at /v1/audio/voices") # Log voice list serving
    return jsonify({"data": available_voices_list})

@app.route('/', methods=['GET'])
def root():
    return jsonify({"message": "Orpheus TTS API is running"})


def list_available_voices(): # Keep this function for command line utility
    """List all available voices with the recommended one marked."""
    print("Available voices (in order of conversational realism):")
    for i, voice in enumerate(AVAILABLE_VOICES):
        marker = "★" if voice == DEFAULT_VOICE else " "
        print(f"{marker} {voice}")
    print(f"\nDefault voice: {DEFAULT_VOICE}")

    print("\nAvailable emotion tags:")
    print("<laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>")


def main():
    parser = argparse.ArgumentParser(description="Orpheus Text-to-Speech API Server")
    parser.add_argument("--list-voices", action="store_true", help="List available voices and exit")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the API server")
    parser.add_argument("--port", type=int, default=5000, help="Port for the API server")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
    parser.add_argument("--api-url-prefix", type=str, default=DEFAULT_API_URL_PREFIX, help=f"API URL prefix (e.g., http://your-lm-studio-host:port), default: {DEFAULT_API_URL_PREFIX}")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help=f"Model name for API payload, default: {DEFAULT_MODEL_NAME}")


    args = parser.parse_args()

    if args.list_voices:
        list_available_voices()
        return

    # Configure Flask app with API URL prefix and model name from arguments
    app.config['API_URL_PREFIX'] = args.api_url_prefix
    app.config['MODEL_NAME'] = args.model

    print("Starting Orpheus TTS API Server...")
    print(f"API URL Prefix: {app.config['API_URL_PREFIX']}")
    print(f"Model Name: {app.config['MODEL_NAME']}")
    print(f"Listening on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=False, processes=1) # important threaded=False and processes=1 for LM Studio


if __name__ == "__main__":
    main()
