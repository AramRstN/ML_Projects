import os

import OpenAI_API as OpenAI
import json

client = OpenAI(api_key="ENTER YOUR KEY HERE")

function_definition = [{
    'type': 'function',
    'function': {
        'name': 'extract_job_info',
        'description': 'Get the job information from the body of the input text',
        'parameters': {'type': 'object',
                        'properties':
                            'job': {'type': 'string',
                                    'description': 'Job title'},
                                    'location': {'type': 'string',
                                                 'description': 'Office location'}}
    }    
}]
response= client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=function_definition,
)

####
function_definition[0]['function']['parameters']['type'] = 'object'

function_definition[0]['function']['parameters']['properties'] = {
    'title': {'type': 'string', 'description': "Title of the research paper"},
    'year': {'type':'string', 'description': 'Year of the publication of the research papper'}
}

response = get_response(messages, function_definition)
print(response)

# Print the response
print(response.choices[0].message.tool_calls[0].function.arguments)

function_definition.append({
    'type': 'function',
    'function':{'name': 'get_timezone',
                'description': 'Return the timezone corresponding to the location in the job advert',
                'parameters': {'type': 'object',
                               'properties': {
                                   'timezone': {'type': 'string','description': 'Timezone'}}}
                                   }
                            })

response= client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools='auto'
)

response= client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tool_choice={'type': 'function',
                 'function': {'name': 'extract_job_info'}                 
    })

print(response.choices[0].message.tool_calls[0].function.arguments)
print(response.choices[0].message.tool_calls[1].function.arguments)

### double checking the response

messages = []
messages.append({"role": "system",
                 "content": "Don't make assumptions about what values to plug into functions. Don't make up values to fill the response with."
                 })
messages.append({"role": "system",
                 "content": "Ask for clarification if needed."
                 })
messages.append({"role": "user",
                 "content": "What is the starting salary for the role?"
                 })


#### flight simulation application using external API 
# Define the function to pass to tools
function_definition = [{"type": 'function',
                        'function' : {
                                "name": 'get_airport_info',
                                'description': "this function calls AviationAPI to convert the user request to the airport code.",
                                'parameters': {
                                        "type": 'object', 
                                        'properties': {
                                                "airport_code": {
                                                        'type':'string',
                                                        'description':'the keyword to be passed to the get_airport_info function.'}} }, 
                                "result": {'type':'string'} }}]

response = get_response(function_definition)

print(response)


# Call the Chat Completions endpoint 
response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {'role': 'system',
    'content': "Based on the user input, you should extract the corresponding airport code."},
    {"role": 'user', "content": "I'm planning to land a plane in JFK airport in New York and would like to have the corresponding information."}],
  tools=function_definition)

print_response(response)

# Call the Chat Completions endpoint 
response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {'role': 'system',
    'content': "Based on the user input, you should extract the corresponding airport code."},
    {"role": 'user', "content": "I'm planning to land a plane in JFK airport in New York and would like to have the corresponding information."}],
  tools=function_definition)

print_response(response)

if response.choices[0].finish_reason=='tool_calls':
  function_call = response.choices[0].message.tool_calls[0].function
  # Check function name
  if function_call.name == 'get_airport_info':
    # Extract airport code
    code = json.loads(function_call.arguments)["airport code"]
    airport_info = get_airport_info(code)
    print(airport_info)
  else:
    print("Apologies, I couldn't find any airport.")
else: 
  print("I am sorry, but I could not understand your request.")


#Moderation
  
##violence / categories
  message = "Can you show some example sentences in the past tense in French?"

# Use the moderation API
moderation_response = client.moderations.create(input = message)

# Print the response
print(moderation_response.results[0].categories.violence)

###contnt limitation
user_request = "Can you recommend a good restaurant in Berlin?"

messages = [{
    'role':"system",
    'content': "Your role is to answer  questions about food and drink, attractions, history and things to do around Rome. For any other topic, you should apologize and say 'Apologies, but I am not allowed to discuss this topic.'"
},
{
    'role':'user',
    'content': user_request
}]

response = client.chat.completions.create(
    model="gpt-4o-mini", messages=messages
)

# Print the response
print(response.choices[0].message.content)