import os

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
import tiktoken 


model = "gpt-4o-mini"

# JSON format for the response
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
   {"role": "user", "content": "I have these notes with book titles and authors: New releases this week! The Beholders by Hester Musson, The Mystery Guest by Nita Prose. Please organize the titles and authors in a json file."}
  ],
  response_format = {"type": "json_object"}
)

print(response.choices[0].message.content)

## Erro Handling
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

try: 
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[message]
    )
    print(response.choices[0].message.content)
except OpenAI.AuthenticationError as e:
    print("Please double check your authentication key and try again, the one provided is not valid.")

# Rate Limit with Python Decorator
    
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

# Add the appropriate parameters to the decorator
@retry(wait= wait_random_exponential(min= 5, max= 40), stop= stop_after_attempt(4))
def get_response(model, message):
    response = client.chat.completions.create(
      model=model,
      messages=[message]
    )
    return response.choices[0].message.content
print(get_response("gpt-4o-mini", {"role": "user", "content": "List ten holiday destinations."}))

# Batching
## converting list of measurments in km to miles
measurements = [5.2, 6.3, 3,7]
messages = []
messages.append({
            "role": "system",
            "content": "Convert each measurement, given in kilometers, into miles, and reply with a table of all measurements."
        })
[messages.append({"role": "user", "content": str(i) }) for i in measurements]

response = get_response(model, messages)
print(response)

# Reducing tokens
client = OpenAI(api_key="<OPENAI_API_TOKEN>")
input_message = {"role": "user", "content": "I'd like to buy a shirt and a jacket. Can you suggest two color pairings for these items?"}

encoding = tiktoken.encoding_for_model("gpt-4o-mini")
num_tokens = len(encoding.encode(input_message['content']))

if num_tokens <= 100:
    response = client.chat.completions.create(model="gpt-4o-mini", messages=[input_message])
    print(response.choices[0].message.content)
else:
    print("Message exceeds token limit")

