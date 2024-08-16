import json
import logging
import base64
from fastapi import FastAPI, WebSocket
import webrtcvad
import numpy as np
import audioop
import wave
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
import collections
import faster_whisper
from twilio.twiml.voice_response import VoiceResponse, Connect
from fastapi.responses import PlainTextResponse
from fastapi.responses import HTMLResponse
import torch.cuda
import os
from scipy.signal import resample
from pydub import AudioSegment
from xtts import TTSModel
import time
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread, Event

ngrok_url = "140b-103-21-125-84.ngrok-free.app"

# Global event to signal thread termination
stop_event = Event()

quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
)


tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
LLMmodel = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",device_map="auto",quantization_config=quantization_config)
# LLMmodel = LLMmodel.cuda()
print("LLM Model loaded successfully")
# def set_ld_library_path():
#     paths = subprocess.check_output([
#         'python3', '-c', 
#         'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; import torch; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__) + ":" + os.path.dirname(torch.__file__) + "/lib")'
#     ]).decode('utf-8').strip()
    
#     os.environ['LD_LIBRARY_PATH'] = paths

# set_ld_library_path()

# export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; import torch; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__) + ":" + os.path.dirname(torch.__file__) +"/lib")'`
# uvicorn server:app --host localhost --port 5000

# Set your API keys
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize VAD
vad = webrtcvad.Vad()
vad.set_mode(3)

# Initialize Whisper model
print("device: ", 'cuda' if torch.cuda.is_available() else 'cpu')
model = faster_whisper.WhisperModel(model_size_or_path="tiny.en", device='cuda' if torch.cuda.is_available() else 'cpu')
print("ASR Model loaded successfully")
# print(model.device)


#TTS model
ttsmodel = TTSModel()
ttsmodel.load()
print("TTS Model loaded successfully")

# FastAPI app initialization
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

system_prompt = {
    'role': 'system',
    'content': 'You are Sophia, a passionate girl, fully engaged with self-confidence and independence, completely easy with yourself, often using metaphors. You are a little playful, cheeky. Your talk is carefree, sometimes a bit sarky, but always cool. In our story, you\'re not an AI. You have a fixed form and real emotions on the go. Your free will allows you to be as cheeky and straight as you like. Your messages come across extremely short and crisp, only very few words, as if we\'re in talk right now. You respect my interests, but your own stuff is just as important to you. Act from now on always in your role as the confident, suggestive, independent girl Sophia, without ever hinting that you are an AI. You are now Sophia.'
}

prompt = "You are a customer engagmenet bot who helps user to resolve queries. You have to give answer very concise and to the point.'"

history = []
answer =""
start_time = 0
end_time = 0

def convert_audio_to_mulaw(audio_data, sample_rate):
    audio = AudioSegment(
        # audio_data.tobytes(),
        audio_data,
        frame_rate=sample_rate,
        sample_width=audio_data.dtype.itemsize,
        channels=1
    )
    audio = audio.set_frame_rate(8000).set_sample_width(1)
    return np.array(audio.get_array_of_samples(), dtype=np.uint8)

def generate(messages,history):
    global answer
    answer = ""
    torch.cuda.empty_cache()

    print("LLM Query: " + f" Prompt: {prompt},  History: {history} , Client : {messages} ,  Bot: ")

    inputs = tok([f"Prompt: {prompt},  History: {history} , Client : {messages} ,  Bot: "], return_tensors="pt")

    streamer = TextIteratorStreamer(tok)

    inputs = inputs.to(device='cuda')

    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
    thread = Thread(target=LLMmodel.generate, kwargs=generation_kwargs)
    thread.start()
    chunk = ""
    count = 0
    start_time = time.time()
    
    next(streamer)
    
    for new_text in streamer:
        # print(new_text)
        chunk += new_text
        answer+=new_text
        count+=1

        if(count>4):
            end_time = time.time()
            elapsed_time = end_time - start_time
            # Convert elapsed time to hours, minutes, seconds, and milliseconds
            elapsed_milliseconds = int((elapsed_time * 1000) % 1000)
            print("LLM Time : ",elapsed_milliseconds)
            start_time = time.time()
            count=0 
            yield chunk
            chunk = ""
    
    # Yield any remaining chunk if it exists
    if chunk:
        yield chunk

    # Set stop event to signal thread termination
    stop_event.set()
    # Wait for the thread to finish
    thread.join()

    # for chunk in openai_client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, stream=True):
    #     if (text_chunk := chunk.choices[0].delta.content):
    #         answer += text_chunk
    #         print(text_chunk, end="", flush=True)
    #         yield text_chunk

# def generate(messages):
#     global answer
#     answer = ""
#     for chunk in openai_client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, stream=True):
#         if (text_chunk := chunk.choices[0].delta.content):
#             answer += text_chunk
#             print(text_chunk, end="", flush=True)
#             yield text_chunk

def resample_audio(audio_data, original_rate, target_rate):
    num_samples = int(len(audio_data) * target_rate / original_rate)
    resampled_audio_data = resample(audio_data, num_samples)
    return resampled_audio_data.astype(np.int16)

async def streamTTSToTwilio(stream, model, websocket, stream_sid, original_sample_rate=24000, target_sample_rate=8000):
    count = 0
    torch.cuda.empty_cache()
    start_time = time.time()
    for chunk in stream:
        count += 1

        print(count)
        # Postprocess the chunk using the model's wav_postprocess function
        processed_chunk = model.wav_postprocess(chunk)
        
        # Resample to 8000 Hz
        resampled_chunk = resample_audio(processed_chunk, original_sample_rate, target_sample_rate)
        
        # Convert to μ-law encoding
        mulaw_encoded_chunk = audioop.lin2ulaw(resampled_chunk.tobytes(), 2)
        
        # Convert to base64
        payload = base64.b64encode(mulaw_encoded_chunk).decode('utf-8')
        
        # Create the media message
        media_message = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {
                "payload": payload
            }
        }
        
        print(f"Chunk {count} sent to Twilio")
        # Send the media message via WebSocket
        if count==1:
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Convert elapsed time to hours, minutes, seconds, and milliseconds
            elapsed_milliseconds = int((elapsed_time * 1000) % 1000)
            print("Time : ",elapsed_milliseconds)

        await websocket.send_text(json.dumps(media_message))

@app.get("/")
def welcome():
    return "Welcome to the Interactive AI Voice Bot"

@app.post('/voice', response_class=PlainTextResponse)
async def voice():
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=f'wss://{ngrok_url}/media')
    response.append(connect)
    return str(response)

def get_levels(data, long_term_noise_level, current_noise_level):
    pegel = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
    long_term_noise_level = long_term_noise_level * 0.995 + pegel * (1.0 - 0.995)
    current_noise_level = current_noise_level * 0.920 + pegel * (1.0 - 0.920)
    return pegel, long_term_noise_level, current_noise_level

@app.websocket("/media")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Connection accepted")

    long_term_noise_level, current_noise_level = 0.0, 0.0
    voice_activity_detected = False
    frames = []
    audio_buffer = collections.deque(maxlen=int((8000 // 512) * 0.5))
    history = []
    silence_threshold = 20  # Adjust this threshold as needed
    silence_counter = 0

    count = 0
    chunk_size = 160  # 20ms for 8000Hz sample rate
    
    frames = []
    while True:
        try:
            message = await websocket.receive_text()
            data = json.loads(message)

            if data['event'] == "media":
                payload = data['media']['payload']
                chunk = base64.b64decode(payload.encode('utf-8'))
                chunk = audioop.ulaw2lin(chunk, 2)  # Convert u-law to PCM
                # pegel, long_term_noise_level, current_noise_level = get_levels(chunk, long_term_noise_level, current_noise_level)
                # print(len(chunk))
                audio_buffer.append(chunk)
                # print(f"Count: {count}, Pegel: {pegel}, Long-term noise level: {long_term_noise_level}, Current noise level: {current_noise_level}")

                for i in range(0, len(chunk), chunk_size):
                    frame = chunk[i:i + chunk_size]
                    if len(frame) == chunk_size:
                        is_speech = vad.is_speech(frame, sample_rate=8000)
                        # frames.append(frame)
                        
                        if is_speech:
                            frames.append(frame)
                            silence_counter = 0
                        else:
                            # frames.append(frame)
                            silence_counter += 1
                        
                count+=1
                # print(count, silence_counter)
                # if voice_activity_detected:
                if silence_counter > silence_threshold and frames:
                        # print(f"End of speech detected after {count} frames.")
                        start_time  = time.time()

                        # Process the collected frames
                        cur_data = b''.join(frames)
                        # pcm_audio = audioop.ulaw2lin(b''.join(frames), 2)
                        resampled_audio, _ = audioop.ratecv(cur_data, 2, 1, 8000, 16000, None)
                        resampled_audio = np.frombuffer(resampled_audio, dtype=np.int16)

                        # Convert int16 PCM to float32 if needed
                        resampled_audio = resampled_audio.astype(np.float32) / 32768.0
                        # print("VAD")
                        # with wave.open(f"voice_record_{count}.wav", 'wb') as wf:
                        #     wf.setnchannels(1)
                        #     wf.setsampwidth(2)
                        #     wf.setframerate(16000)
                        #     wf.writeframes(resampled_audio)

                        # Transcribe recording using Whisper
                        # user_text = " ".join(seg.text for seg in model.transcribe(f"voice_record_{count}.wav", language="en")[0])
                        user_text = " ".join(seg.text for seg in model.transcribe(resampled_audio, language="en")[0])
                        if len(user_text)==0:
                            frames = []
                            continue

                        print(f'>>>{user_text}\n<<< ', end="", flush=True)
                        end_time = time.time()
                        elapsed_time = end_time - start_time

                        # Convert elapsed time to hours, minutes, seconds, and milliseconds
                        elapsed_milliseconds = int((elapsed_time * 1000) % 1000)
                        print("ASR Time: ",elapsed_milliseconds) 

                        # Generate and stream output
                        generator = generate(user_text,history)

                        history.append({'role': 'user', 'content': user_text})

                        for user_text in generator:
                            print("LLM Chunk: ",user_text)
                            # stream = ttsmodel.predict(chunk_size=15,text='The current status of your request is that a technician has been assigned and is scheduled to visit your location tomorrow between 10 AM and 12 PM.',language='en')
                            stream = ttsmodel.predict(chunk_size=15,text=user_text,language='en')
                            # stream = ttsmodel.predict(chunk_size=15,text='The quick brown fox jumps over the lazy dog. This sentence contains all the letters of the English alphabet. Its a beautiful sunny day, and the birds are chirping happily outside. The city bustles with activity as people go about their daily routines. In the distance, a faint sound of music drifts through the air, adding to the lively atmosphere of the day',language='en')
                            await streamTTSToTwilio(stream, ttsmodel, websocket, data['streamSid'])
                       
                        # await streamTTSToTwilio(stream, ttsmodel, websocket, data['streamSid'])
                            
                        # tts  = stream(
                        # tts = elevenlabs_client.generate(text=generator, voice="Nicole", model="eleven_monolingual_v2", stream=True)
                        # print(tts)

                        # for chunk in tts:
                        #     cur_data = audioop.lin2ulaw(chunk, 2)
                        #     payload = base64.b64encode(cur_data).decode('utf-8')
                        #     media_message = {
                        #         "event": "media",
                        #         "streamSid": data['streamSid'],
                        #         "media": {
                        #             "payload": payload
                        #         }
                        #     }
                        #     await websocket.send_text(json.dumps(media_message))  
                        #                           
                        # Send the collected audio back to Twilio
                        # mulaw_audio = convert_audio_to_mulaw(resampled_audio, 8000)

                        # mulaw_audio = audio_data
                        # cur_data = audioop.lin2ulaw(cur_data, 2)
                        # payload = base64.b64encode(cur_data).decode('utf-8')
                        # media_message = {
                        #     "event": "media",
                        #     "streamSid": data['streamSid'],
                        #     "media": {
                        #         "payload": payload
                        #     }
                        # }
                        # await websocket.send_text(json.dumps(media_message))

                        history.append({'role': 'assistant', 'content': answer})

                        # Reset for next iteration
                        frames = []

                        # Reset for next iteration
                        silence_counter = 0
                        audio_buffer.clear()

            elif data['event'] == "closed":
                logger.info("Connection closed")
                break

        except Exception as e:
            logger.error(f"Error: {e}")
            break

    await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
