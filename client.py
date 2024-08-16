import asyncio
import base64
import json
import websockets
import wave
import audioop

async def send_audio(file_path):
    uri = "ws://localhost:5000/media"  # Replace with your actual WebSocket server URI

    # Open the audio file
    with wave.open(file_path, 'rb') as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 8000

        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket server")
            try:
                while True:
                    data = wf.readframes(320)  # Read 20ms of audio data at 8000 Hz, 16-bit PCM
                    if not data:
                        break
                    
                     # Convert PCM data to μ-law
                    data = audioop.lin2ulaw(data, 2)

                    # Encode the audio data as base64
                    payload = base64.b64encode(data).decode('utf-8')
                    message = json.dumps({
                        "event": "media",
                        "streamSid": "dummyStreamSid",  # Simulate a stream ID
                        "media": { 
                            "payload": payload
                        }
                    })
                    await websocket.send(message)
                    await asyncio.sleep(0.02)  # Simulate real-time streaming by waiting 20ms per frame

                # Send the closed event to indicate the end of the stream
                await websocket.send(json.dumps({"event": "closed"}))
                print("Audio stream sent successfully")

            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    # import sys
    # if len(sys.argv) != 2:
    #     print("Usage: python dummy_twilio_client.py <path_to_audio_file>")
    #     sys.exit(1)

    audio_file_path = "format_recording.wav"
    asyncio.run(send_audio(audio_file_path))
