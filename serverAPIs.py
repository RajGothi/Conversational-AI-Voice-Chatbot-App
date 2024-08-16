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
import asyncio
import websockets
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import requests

ngrok_url = "7f53-103-21-126-80.ngrok-free.app"

# Global event to signal thread termination
stop_event = Event()

# quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.float16,
# )


# tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# LLMmodel = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",device_map="auto",quantization_config=quantization_config)
# # LLMmodel = LLMmodel.cuda()
# print("LLM Model loaded successfully")

# export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; import torch; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__) + ":" + os.path.dirname(torch.__file__) +"/lib")'`
# uvicorn server:app --host localhost --port 5000

# Set your API keys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# elevenlabs_client = ElevenLabs(api_key="sk_0f4cb41e8d3c91586950d9316d03b4b7804dcc574f30d9a4")

# groq API key : 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
# deepgram API : 24a194609069414f5ab3e0657c56f13000eb38ae


# Initialize VAD
vad = webrtcvad.Vad()
vad.set_mode(3)

# Initialize Whisper model
# print("device: ", 'cuda' if torch.cuda.is_available() else 'cpu')
# model = faster_whisper.WhisperModel(model_size_or_path="tiny.en", device='cuda' if torch.cuda.is_available() else 'cpu')
# print("ASR Model loaded successfully")
# print(model.device)


#TTS model
# ttsmodel = TTSModel()
# ttsmodel.load()
# print("TTS Model loaded successfully")

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


load_dotenv()


class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", groq_api_key="ABC")
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview", openai_api_key=os.getenv("OPENAI_API_KEY"))
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125", openai_api_key=os.getenv("OPENAI_API_KEY"))

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load the system prompt from a file
        with open('system_prompt.txt', 'r') as file:
            system_prompt = file.read().strip()
        
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])
        self.response = ""

    def process(self, text):
        self.response = ""
        self.memory.chat_memory.add_user_message(text)  # Add user message to memory

        # start_time = time.time()

        # response = ""
        chain = self.prompt | self.llm 
        count = 0
        segment = ""
        # Go get the response from the LLM
        for chunk in chain.stream({"text": text,'chat_history':self.memory.chat_memory.messages}):
            # print(chunk.content)
            count += 1
            self.response += chunk.content
            segment += chunk.content
            if count==5:
                count = 0
                print(segment)
                temp = segment
                segment = ""
                yield temp

    # In future we will use...
    def segment_text_by_sentence(text):
        import re
        sentence_boundaries = re.finditer(r'(?<=[.!?])\s+', text)
        boundaries_indices = [boundary.start() for boundary in sentence_boundaries]
        
        segments = []
        start = 0
        for boundary_index in boundaries_indices:
            segments.append(text[start:boundary_index + 1].strip())
            start = boundary_index + 1
        segments.append(text[start:].strip())

        return segments
                    
            # print(chunk, end="|", flush=True)   

        # response = self.conversation.invoke({"text": text})
        
        # end_time = time.time()

        # self.memory.chat_memory.add_ai_message(response)  # Add AI response to memory

        # elapsed_time = int((end_time - start_time) * 1000)
        # print(f"LLM ({elapsed_time}ms): {response}")
        # return response

# def generate(messages,history):
#     global answer
#     answer = ""
#     torch.cuda.empty_cache()

#     print("LLM Query: " + f" Prompt: {prompt},  History: {history} , Client : {messages} ,  Bot: ")

#     inputs = tok([f"Prompt: {prompt},  History: {history} , Client : {messages} ,  Bot: "], return_tensors="pt")

#     streamer = TextIteratorStreamer(tok)

#     inputs = inputs.to(device='cuda')

#     generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
#     thread = Thread(target=LLMmodel.generate, kwargs=generation_kwargs)
#     thread.start()
#     chunk = ""
#     count = 0
#     start_time = time.time()
    
#     next(streamer)
    
#     for new_text in streamer:
#         # print(new_text)
#         chunk += new_text
#         answer+=new_text
#         count+=1

#         if(count>4):
#             end_time = time.time()
#             elapsed_time = end_time - start_time
#             # Convert elapsed time to hours, minutes, seconds, and milliseconds
#             elapsed_milliseconds = int((elapsed_time * 1000) % 1000)
#             print("LLM Time : ",elapsed_milliseconds)
#             start_time = time.time()
#             count=0 
#             yield chunk
#             chunk = ""
    
#     # Yield any remaining chunk if it exists
#     if chunk:
#         yield chunk

#     # Set stop event to signal thread termination
#     stop_event.set()
#     # Wait for the thread to finish
#     thread.join()

#     # for chunk in openai_client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, stream=True):
#     #     if (text_chunk := chunk.choices[0].delta.content):
#     #         answer += text_chunk
#     #         print(text_chunk, end="", flush=True)
#     #         yield text_chunk

# # def generate(messages):
# #     global answer
# #     answer = ""
# #     for chunk in openai_client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, stream=True):
# #         if (text_chunk := chunk.choices[0].delta.content):
# #             answer += text_chunk
# #             print(text_chunk, end="", flush=True)
# #             yield text_chunk

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

class TTSElevenLab:
    def __init__(self,websocket,stream_sid):
        self.ELEVENLABS_API_KEY = 'sk_'
        self.VOICE_ID = '21m00Tcm4TlvDq8ikWAM'
        self.model_name = "eleven_turbo_v2_5"
        self.websocket = websocket
        self.stream_sid = stream_sid
    
    async def text_to_speech_input_streaming(self, text_iterator):

        """Send text to ElevenLabs API and stream the returned audio."""
        
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.VOICE_ID}/stream-input?model_id={self.model_name}"

        async with websockets.connect(uri) as websocket:

            await websocket.send(json.dumps({
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                "xi_api_key": self.ELEVENLABS_API_KEY,
            }))

            async def listen():
                """Listen to the websocket for audio data and stream it."""

                # payloadmsg = base64.b64encode(chunk).decode('utf-8')

                # print(payloadmsg)
                # Create the media message
                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        if data.get("audio"):
                            payloadmsg = base64.b64decode(data["audio"])
                            media_message = {
                                                "event": "media",
                                                "streamSid": self.stream_sid,
                                                "media": {
                                                    "payload":payloadmsg 
                                                }
                                            }
                            # print("media message",media_message)
                            # send the TTS audio to the attached phonecall
                            await self.websocket.send_text(json.dumps(media_message))

                        elif data.get('isFinal'):
                            break
                    except websockets.exceptions.ConnectionClosed:
                        print("Connection closed")
                        break

            listen_task = asyncio.create_task(listen())

            for text in text_iterator:
                print("LLM chunk ",text)
                await websocket.send(json.dumps({"text": text, "try_trigger_generation": True}))

            await websocket.send(json.dumps({"text": ""}))

            await listen_task

    # async def process():
    #     text_to_speech_input_streaming(VOICE_ID, text_iterator())


class TTSDeepgram:
    def __init__(self,websocket,stream_sid):
        self.DEEPGRAM_URL = "https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=mulaw&sample_rate=8000&container=none"
        DEEPGRAM_API_KEY = 'XAS'
        
        self.headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json"
        }
        self.websocket = websocket
        self.stream_sid = stream_sid

    async def process(self,text):
        payload = {
            "text": text
        }
        response = requests.post(self.DEEPGRAM_URL, headers=self.headers, json=payload, stream=True)
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                # chunk is rawmulaw
                payloadmsg = base64.b64encode(chunk).decode('utf-8')

                # print(payloadmsg)
                # Create the media message
                media_message = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {
                        "payload":payloadmsg 
                    }
                }
                # print("media message",media_message)
                # send the TTS audio to the attached phonecall
                await self.websocket.send_text(json.dumps(media_message))
                # print("reach here")

llm = LanguageModelProcessor()

async def get_answer(transcript, websocket, stream_sid):

    stream = llm.process(transcript) #generator which give chunk wise text

    TTSStream = TTSDeepgram(websocket,stream_sid)

    # TTSStreamElevenLab  = TTSElevenLab(websocket,stream_sid)

    # uncomment below for Local TTS and deeepgram TTS
    for chunk in stream:
        print("LLM Chunk: ",chunk)

        #For the Local TTS...
        #pass this chunk text to TTS model,which will Stream TTS to twilio....
        # stream = ttsmodel.predict(chunk_size=15,text=chunk,language='en')
                            # stream = ttsmodel.predict(chunk_size=15,text='The quick brown fox jumps over the lazy dog. This sentence contains all the letters of the English alphabet. Its a beautiful sunny day, and the birds are chirping happily outside. The city bustles with activity as people go about their daily routines. In the distance, a faint sound of music drifts through the air, adding to the lively atmosphere of the day',language='en')
        # await streamTTSToTwilio(stream, ttsmodel, websocket,stream_sid)

        # For the Deepgram TTS....
        await TTSStream.process(chunk)
    

    #Here write for the Elevenlabs...
    # await TTSStreamElevenLab.text_to_speech_input_streaming(stream)


    llm.memory.chat_memory.add_ai_message(llm.response)  # Add AI response to memory

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

# Replace with your Deepgram API key
DEEPGRAM_API_KEY = '24'

def deepgram_connect():
    extra_headers = {
        'Authorization': f'Token {DEEPGRAM_API_KEY}'
    }
    return websockets.connect("wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000&endpointing=true", extra_headers=extra_headers)
    # return websockets.connect("wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000&endpointing=200", extra_headers=extra_headers)
    # return websockets.connect("wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=8000&endpointing=true", extra_headers=extra_headers)
    
@app.websocket("/media")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Connection accepted")

    frames = b''
    # audio_buffer = collections.deque(maxlen=int((8000 // 512) * 0.5))
    
    silence_threshold = 30  # Adjust this threshold as needed
    silence_counter = 0
    transcript = ""

    stream_sid = 0
    # sil_only = 1

    count = 0
    chunk_size = 160  # 20ms for 8000Hz sample rate

    outbox = asyncio.Queue()
    transcript_event = asyncio.Event()

    async def client_receiver():
        nonlocal silence_counter, frames, count, transcript, stream_sid
        BUFFER_SIZE = 20 * 160
        # BUFFER_SIZE = 3000
        # buffer = bytearray(b'')
        empty_byte_received = False

        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                # print("received")
                if data['event'] == "media":
                    payload = data['media']['payload']
                    chunk = base64.b64decode(payload.encode('utf-8'))
                    frames += chunk
                    stream_sid = data['streamSid']
                    # frames.append(chunk)

                    if len(frames) >= BUFFER_SIZE or empty_byte_received:
                        print("Insideeeeeeeeeeeeee")
                        # cur_data = b''.join(frames)
                        # cur_data = frames
                        outbox.put_nowait(frames)
                        # buffer = bytearray(b'')
                        frames = b'' # Reset frames after processing

                        # await transcript_event.wait()
                        # transcript_event.clear()
                    
                    # chunk = audioop.ulaw2lin(chunk, 2)  # Convert u-law to PCM
                    # audio_buffer.append(chunk)

                    # for i in range(0, len(chunk), chunk_size):
                    #     frame = chunk[i:i + chunk_size]
                    #     is_speech = vad.is_speech(frame, sample_rate=8000)

                    #     if is_speech:
                    #         # frames += frame
                    #         # sil_only = 0
                    #         frames.append(frame)
                    #         silence_counter = 0
                    #     else:
                    #         # frames += frame
                    #         frames.append(frame)
                    #         silence_counter += 1

                    # count += 1
                    # if silence_counter > silence_threshold and len(frames)!=0:

                    #     silence_counter = 0
                    #     # start_time = time.time()
                    #     cur_data = b''.join(frames)
                    #     # cur_data = frames
                    #     # resampled_audio, _ = audioop.ratecv(cur_data, 2, 1, 8000, 16000, None)
                    #     # resampled_audio = np.frombuffer(resampled_audio, dtype=np.int16)
                    #     # resampled_audio = resampled_audio.astype(np.float32) / 32768.0


                    #     # Put the audio chunk in the outbox queue to send to Deepgram
                    #     print("sent from VAD")
                    #     outbox.put_nowait(cur_data)
                    #     frames = []  # Reset frames after processing

                    #     # # Wait for Deepgram to finish processing
                    #     # await transcript_event.wait()
                    #     # transcript_event.clear()

                    #     # Wait for Deepgram to finish processing
                    #     # try:
                    #     #     await asyncio.wait_for(transcript_event.wait(), timeout=1.0)
                    #     # except asyncio.TimeoutError:
                    #     #     pass

                    #     # transcript_event.clear()

                    #     #collect all the transcript from deepgram and give to groq...
                    
                    #     if len(transcript.strip())!=0:

                    #         print("VAD : ",transcript)
                    #         await get_answer(transcript, websocket, data['streamSid'])
                    #         transcript = ""
                    #         # pass
                    #         #pass this transcript to the groq for streaming which will give answer... write some asynchoronous function which we will run without waiting....                        
                    #         # streaming from the groq tokens pass to deepgram TTS... 

                    #     transcript = ""

                    # elif len(frames) >= BUFFER_SIZE or empty_byte_received:
                    #     print("Insideeeeeeeeeeeeee")
                    #     cur_data = b''.join(frames)
                    #     # cur_data = frames
                    #     outbox.put_nowait(cur_data)
                    #     # buffer = bytearray(b'')
                    #     frames = [] # Reset frames after processing

            except Exception as e:
                logger.error(f"Error in client_receiver: {e}")
                break

        outbox.put_nowait(b'')

    async def deepgram_sender(deepgram_ws):
        while True:
            chunk = await outbox.get()
            print("deepgram send")
            # print(chunk)
            await deepgram_ws.send(chunk)

    async def deepgram_receiver(deepgram_ws):
        nonlocal transcript
        async for message in deepgram_ws:
            try:
                print("deepgram received")
                res = json.loads(message)
                # print(res)
                transcript_chunk = (
                    res.get("channel", {})
                    .get("alternatives", [{}])[0]
                    .get("transcript", "")
                )

                # handle local server messages, if we're streaming to our local server
                if res.get("msg"):
                    print(f"Server message: {res.get('msg')}")
                elif transcript_chunk:
                    transcript += " " + transcript_chunk
                    print(f"DG transcript: {transcript_chunk}")

                # Check if the message indicates the end of the processing
                if res.get("speech_final"):
                    print("Endpoint: ",transcript)
                    # transcript = ""
                    transcript_event.set()
            except:
                logger.error('Failed to parse Deepgram response as JSON')
    
    async def handle_transcription():
        nonlocal transcript,stream_sid
        while True:
            await transcript_event.wait()
            # Pass the transcript to the LLM
            # print(f"Transcription to be sent to LLM: {transcript}")

            # Call your LLM processing function here, e.g.,
            # result = await process_with_llm(transcript)
            # Reset the transcript_event for future use

            await get_answer(transcript, websocket, stream_sid)

            transcript_event.clear()
            # Clear the transcript after processing
            transcript = ""


    async with deepgram_connect() as deepgram_ws:
        await asyncio.gather(
            asyncio.create_task(client_receiver()),
            asyncio.create_task(deepgram_sender(deepgram_ws)),
            asyncio.create_task(deepgram_receiver(deepgram_ws)),
            asyncio.create_task(handle_transcription())
        )

    await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
