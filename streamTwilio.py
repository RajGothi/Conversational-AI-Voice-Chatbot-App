from twilio.twiml.voice_response import Connect, VoiceResponse, Say, Stream
from twilio.rest import Client

# response = VoiceResponse()
# connect = Connect()
# connect.stream(url='wss://4918-103-21-126-80.ngrok-free.app/media')
# response.append(connect)
# response.say(
#     'This TwiML instruction is unreachable unless the Stream is ended by your WebSocket server.'
# )

# print(response)


def make_call():
    # URL for the TwiML response
    twiml_url = f"{ngrok_url}/voice"

    # twiml_url = url_for('voice', _external=True)
    
    call = client.calls.create(
        # twiml="<Response><Say>Hello, please wait while we process your call.</Say></Response>",
        from_="+18149923665",
        to="+918306869513",
        # twiml = response
        url=twiml_url
    )

    print(f'Call SID: {call.sid}')

ngrok_url = 'https://7f53-103-21-126-80.ngrok-free.app'

client = Client(account_sid, auth_token)

make_call()





# def make_call():    
#     call = client.calls.create(
#         twiml='<Response><Say>Hello, please wait while we process your call.</Say></Response>',
#         from_="+18149923665",
#         to="+918306869513",
#     )

#     print(f'Call SID: {call.sid}')

# account_sid = 'XYZ'
# auth_token = 'ABC'
# client = Client(account_sid, auth_token)

# make_call()


