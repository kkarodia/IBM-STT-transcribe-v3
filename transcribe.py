import argparse
import base64
import configparser
import json
import threading
import time
import os

import pyaudio
import websocket
from websocket._abnf import ABNF
from flask import Flask, render_template_string, Response, jsonify
import queue

app = Flask(__name__)

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
FINALS = []
LAST = None

REGION_MAP = {
    'us-east': 'us-east.speech-to-text.watson.cloud.ibm.com',
    'us-south': 'us-south.speech-to-text.watson.cloud.ibm.com',
    'eu-gb': 'eu-gb.speech-to-text.watson.cloud.ibm.com',
    'eu-de': 'eu-de.speech-to-text.watson.cloud.ibm.com',
    'au-syd': 'au-syd.speech-to-text.watson.cloud.ibm.com',
    'jp-tok': 'jp-tok.speech-to-text.watson.cloud.ibm.com',
}

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transcription App</title>
</head>
<body>
    <h1>Transcription App</h1>
    <button id="startBtn">Start Transcribing</button>
    <button id="stopBtn" disabled>Stop Transcribing</button>
    <div id="transcript"></div>
    <h2>Final Transcript:</h2>
    <div id="finalTranscript"></div>

    <script>
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const transcriptDiv = document.getElementById('transcript');
        const finalTranscriptDiv = document.getElementById('finalTranscript');
        let eventSource;

        startBtn.addEventListener('click', () => {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            transcriptDiv.innerHTML = '';

            eventSource = new EventSource('/start_transcription');
            eventSource.onmessage = (event) => {
                transcriptDiv.innerHTML += event.data + '<br>';
            };
        });

        stopBtn.addEventListener('click', () => {
            stopBtn.disabled = true;
            startBtn.disabled = false;

            fetch('/stop_transcription')
                .then(response => response.json())
                .then(data => {
                    if (eventSource) {
                        eventSource.close();
                    }
                    getFinalTranscript();
                });
        });

        function getFinalTranscript() {
            fetch('/get_final_transcript')
                .then(response => response.json())
                .then(data => {
                    finalTranscriptDiv.innerHTML = data.transcript;
                });
        }
    </script>
</body>
</html>
"""

transcription_queue = queue.Queue()
final_transcript = []
is_transcribing = False

def read_audio(ws):
    global RATE, is_transcribing
    p = pyaudio.PyAudio()
    RATE = int(p.get_default_input_device_info()['defaultSampleRate'])
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")
    while is_transcribing:
        data = stream.read(CHUNK)
        ws.send(data, ABNF.OPCODE_BINARY)

    stream.stop_stream()
    stream.close()
    print("* done recording")

    data = {"action": "stop"}
    ws.send(json.dumps(data).encode('utf8'))
    time.sleep(1)
    ws.close()
    p.terminate()

def on_message(ws, msg):
    global final_transcript
    data = json.loads(msg)
    if "results" in data:
        transcript = data['results'][0]['alternatives'][0]['transcript']
        print(transcript)
        transcription_queue.put(transcript)
        if data["results"][0]["final"]:
            final_transcript.append(transcript)

def on_error(ws, error):
    print(error)

def on_close(ws, close_status_code, close_msg):
    global is_transcribing
    is_transcribing = False
    print("WebSocket closed")
    save_transcript()

def save_transcript():
    global final_transcript
    full_transcript = " ".join(final_transcript)
    with open("transcript.txt", "w") as f:
        f.write(full_transcript)
    print("Transcript saved to transcript.txt")
    final_transcript = []

def on_open(ws):
    global is_transcribing
    is_transcribing = True
    print("WebSocket opened")
    data = {
        "action": "start",
        "content-type": f"audio/l16;rate={RATE}",
        "continuous": True,
        "interim_results": True,
        "word_confidence": True,
        "timestamps": True,
        "max_alternatives": 3
    }
    ws.send(json.dumps(data).encode('utf8'))
    threading.Thread(target=read_audio, args=(ws,)).start()

def get_url():
    host = REGION_MAP["us-south"]
    return (f"wss://api.{host}/instances/57f91e09-9377-4a74-a1d2-43230edf7829/v1/recognize"
            "?model=en-US_BroadbandModel")

def get_auth():
    apikey = "Oogv4QFoAdBHL6kvvwnm-rOXAQfmSQFXvxHXWTGMUqfn"
    return ("apikey", apikey)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/start_transcription')
def start_transcription():
    global is_transcribing
    is_transcribing = True

    def generate():
        headers = {}
        userpass = ":".join(get_auth())
        headers["Authorization"] = "Basic " + base64.b64encode(userpass.encode()).decode()
        url = get_url()

        ws = websocket.WebSocketApp(url,
                                    header=headers,
                                    on_message=on_message,
                                    on_error=on_error,
                                    on_close=on_close)
        ws.on_open = on_open
        
        threading.Thread(target=ws.run_forever).start()

        while is_transcribing:
            try:
                transcript = transcription_queue.get(timeout=1)
                yield f"data: {transcript}\n\n"
            except queue.Empty:
                pass

    return Response(generate(), mimetype='text/event-stream')

@app.route('/stop_transcription')
def stop_transcription():
    global is_transcribing
    is_transcribing = False
    return jsonify({"status": "Transcription stopped"})

@app.route('/get_final_transcript')
def get_final_transcript():
    if os.path.exists("transcript.txt"):
        with open("transcript.txt", "r") as f:
            transcript = f.read()
        return jsonify({"transcript": transcript})
    else:
        return jsonify({"transcript": "No transcript available."})

if __name__ == "__main__":
    app.run(debug=True, port=8080)