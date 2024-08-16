# Conversational AI Voice Chatbot App

## Overview

The Conversational AI Voice Chatbot App is a voice-enabled chatbot application that allows users to interact via phone calls and query product-related issues. Developed during the summer of 2024, this project integrates various APIs to manage phone calls, process natural language, and handle speech synthesis.

## Features

- **Voice Interaction**: Users can call in and interact with the chatbot to resolve product-related issues.
- **API Integrations**:
  - **Twilio API**: Manages phone call routing and interactions.
  - **OpenAI API**: Provides natural language processing.
  - **Deepgram API**: Converts speech to text.
  - **ElevenLab API**: Converts text to speech with customizable voice options.
- **Dynamic Language Models**: Language models can be switched dynamically via Langchain.
- **Customizable Voice Options**: Choose between male and female voices using ElevenLab.
- **Response Latency**: Achieved a response latency of 600ms from the client's stop speaking to the bot’s first response.

## Technologies Used

- **Programming Languages**: Python, JavaScript
- **Frameworks and Libraries**:
  - **FASTAPI**: For building the API server.
  - **Twilio**: For phone call management.
  - **OpenAI**: For natural language processing.
  - **Deepgram**: For speech-to-text conversion.
  - **ElevenLab**: For text-to-speech conversion.
  - **Langchain**: For dynamic language model switching.

## Setup Instructions

### Prerequisites

- Python 3.x
- Node.js and npm
- Accounts with Twilio, OpenAI, Deepgram, and ElevenLab

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd <repository-directory>

2. **Create account on each platform and add API key:**
```bash
TWILIO_ACCOUNT_SID=<your-twilio-account-sid>
TWILIO_AUTH_TOKEN=<your-twilio-auth-token>
OPENAI_API_KEY=<your-openai-api-key>
DEEPGRAM_API_KEY=<your-deepgram-api-key>
ELEVENLAB_API_KEY=<your-elevenlab-api-key>
```

3. Run the Server
```bash
uvicorn main:app --reload
```

## Usage

Make a Phone Call: Dial the provided Twilio number to interact with the chatbot.

Query Product Issues: Ask questions or describe issues related to the product during the call.

Voice Customization: Choose between available voice options as per your preference.

Run client file, you will get a call to your registered phone.
