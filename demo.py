from twilio.rest import Client
 
# Your Account SID and Auth Token from twilio.com/console

client = Client(account_sid, auth_token)
 
# The Twilio number you are calling from
from_number = '+18149923665'
# The number you are calling to
to_number = '+918306869513'
 
# The URL that returns TwiML instructions for the call
twiml_url = 'http://demo.twilio.com/docs/voice.xml'
 
call = client.calls.create(
    to=to_number,
    from_=from_number,
    url=twiml_url
)
 
print(f"Call initiated with SID: {call.sid}")