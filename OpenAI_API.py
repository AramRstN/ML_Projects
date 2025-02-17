import os

from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI"])

# Q&A
response = client.chat.completions.create(
    model="gpt-4o-mini",
    # Write your prompt
    messages=[{"role": "user", "content": "How old is the GPT?"}],
    max_tokens=200
)

print(response.choices[0].message.content)

# Prompt completion
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Life is like a box of chocolates."}],
    temperature = 2
)
print(response.choices[0].message.content)

#content transformation

prompt = """Update name to Maarten,
pronouns to he/him, and job title to Senior Content
Developerin the following text:

Joanne is a Content Developer at DataCamp. Her favorite programming language is R,
which she uses for her statistical analyses."""


response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
print(response.choices[0].message.content)

# Content Generation:

prompt = "Create a tagline for a new hot dog stand."
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=100
)
print(response.choices[0].message.content)

# Categorizing

#default
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Classify the following animals into    categories: zebra, crocodile, blue whale, polar bear, salmon, dog."}],
    max_tokens=50
)
print(response.choices[0].message.content)

#Specific
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Classify the following animals into animals    with fur and without: zebra, crocodile, dolphin, polar bear, salmon, dog."}],
    max_tokens=50
)
print(response.choices[0].message.content)

# sentiment analysis
prompt1 = """Classify sentiment in the following statements:
1. The service was very slow
2. The steak was awfully tasty!
3. Meal was decent, but I've had better.
4. My food was delayed, but drinks were good."""

prompt2 = """Classify sentiment as 1-5 (bad-good) in the following statements:
1. The service was very slow
2. The steak was awfully tasty!
3. Meal was decent, but I've had better.
4. My food was delayed, but drinks were good."""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt1}],
    max_tokens=50
)
print(response.choices[0].message.content)


# zero/one/few -shot prompting

#one-shot
prompt = """Classify sentiment in the following statements:
The service was very slow // Disgruntled
Meal was decent, but I've had better. //"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
print(response.choices[0].message.content)

#few-shot

prompt = """Classify sentiment in the following statements:
The service was very slow // Disgruntled
The steak was awfully tasty! // Delighted
Good experience overall. // Satisfied
Meal was decent, but I've had better. // """
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
    )
print(response.choices[0].message.content)

#roles
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system",
               "content": "You are a data science tutor who speaks concisely."},
               {"role": "user",
                "content": "What is the difference between mutable and immutable objects?"}]
)
print(response.choices[0].message.content)


# Providing Examples:
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system",
               "content": "You are a data science tutor who speaks concisely."},
               {"role": "user",
                "content": "How do you define a Python list?"},
                {"role": "assistant",
                 "content": "Lists are defined by enclosing a comma-separated sequence of objects inside square brackets [ ]."},
                 {"role": "user",
                  "content": "What is the difference between mutable and immutable objects?"}]
)
print(response.choices[0].message.content)


#conversation:
messages = [{"role": "system", "content": "You are a helpful math tutor."}]
user_msgs = ["Explain what pi is.", "Summarize this in two bullet points."]

for q in user_msgs:
    print("User: ", q)
    
    # Create a dictionary for the user message from q and append to messages
    user_dict = {"role": "user", "content": q}
    messages.append(user_dict)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=100
    )
    
    assistant_dict = {"role": "assistant", "content": response.choices[0].message.content}
    messages.append(assistant_dict)
    print("Assistant: ", response.choices[0].message.content, "\n")


# Moderation 
response = client.moderations.create(
    model='text-moderation-latest',
    input="My favorite book is To Kill a Mockingbird."
)

print(response.results[0].category_scores)

#speach to text
audio_file = open("openai-audio.mp3", "rb")
response = client.audio.transcriptions.create(model="whisper-1", file=audio_file)

print(response.text)

audio_file= open("audio.m4a","rb")
response = client.audio.transcriptions.create(model="whisper-1",file=audio_file)

print(response.text)

#Translation
audio_file = open("audio.m4a", "rb")
response = client.audio.translations.create(model="whisper-1", file=audio_file)

print(response.text)


#with prompt translation
audio_file = open("audio.wav","rb")
prompt = "the audio relates to a recent World Bank report"

# Create a translation from the audio file
response = client.audio.translations.create(model="whisper-1",file=audio_file,prompt=prompt)

print(response.text)


## combining models

audio_file = open("audio.wav", "rb")

# Create a transcription request using audio_file
audio_response = client.audio.transcriptions.create(model="whisper-1",file=audio_file)
text=audio_response.text
prompt= "what language is this text:"+text
# Create a request to the API to identify the language spoken
chat_response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{
        "role":"user","content":prompt
    }]
    )
print(chat_response.choices[0].message.content)

##meeting summary
audio_file = open("datacamp-q2-roadmap.mp3","rb")

# Create a transcription request using audio_file
audio_response = client.audio.transcriptions.create(model="whisper-1",file=audio_file)
transcript= audio_response.text

prompt= "summarize this text into xoncise bullet points:" + transcript
# Create a request to the API to summarize the transcript into bullet points
chat_response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages= [{
        "role":"user",
        "content": prompt
    }]
)
print(chat_response.choices[0].message.content)

