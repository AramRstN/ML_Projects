import torch
import evaluate
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer, AutoModelForAudioClassification
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

# using pretrained model / HugingFace

#summarization
summarizer = pipeline(task="summarization", model= "facebook/bart-large-cnn")

text = """Where You Might Need Improvement:
1.Confidence in Presenting Yourself: While you have deep technical skills, you sometimes struggle to highlight them effectively, especially in emails or interviews. You’re hesitant about how to phrase things, especially when discussing your background. You might need to be more assertive in showcasing your strengths.
2.Letting Go of the Past: You seem to carry the weight of past challenges (PhD issues, academic sanctions) quite heavily. While they are real concerns, they shouldn’t define you. A bit more forward-looking optimism might help.
3.Physical Health & Fitness: You’ve recognized this yourself by wanting a weight-loss plan. Given your work is mostly intellectual, finding a sustainable routine for physical well-being could improve your energy and confidence.
4.Speed in Taking Action: You are careful in your decisions, which is good, but sometimes overthinking might slow you down. Trusting your instincts a bit more could help you act faster in situations like job applications or networking."""
summary = summarizer(text, max_length = 50)
# clean_up_tokenization_space = True to remove innecessary white space.

print(summary[0]["summary_text"])

#Generation
generator = pipeline(task="text-generation", model="distilgpt2")

#1
prompt = "the Gion neighborhod in Kyoto is famous for"
output = generator(prompt, max_length=100, pad_token_id=generator.tokenizer.eos_token_id)
#truncation =True input longer than max_length
print(summary[0]["generated_text"])

#2. Guiding the model
review = "This book was great. I enjoyed the plot twist in Chapter 10."
response = "Dear reader, thank you for you review."
prompt = f"Book review:\n{review}\n\nBook shop response to rhe review:\n{response}"
output = generator(prompt, max_length=100, pad_token_id=generator.tokenizer.eos_token_id)
print(output[0]["generatef_text"])

#Translation
translator = pipeline(task="translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
text = "Walking aimd Gion's Machiya wooden houses was memerizing experiance."
output = translator(text, clean_up_tokenization_space= True)
print(output[0]["translation_text"])

#Encoder-only like BERT -> understandin the input (text classificatio, extraxtive QA, Sentiment analysis)
#Decoder-only like GPPT -> focuses on output (text generation, Generative AQ)
#encoder-decoder like -> both input and output (translation, summarization)

question = "Who painted the Mona Lisa?"
# Define the appropriate model (gpt2 or "distilbert-base-uncased-distilled-squad")
qa = pipeline(task="question-answering", model="gpt2")
input_text = f"Context: {text}\n\nQuestion: {question}\n\nAnswer:"
output = qa({"context": text, "question": question}, max_length=150)
print(output['answer'])

# IMDB dataset
train_data = load_dataset("imdb", split="train")
train_data = data.shard(num_shards=4, index=0)
test_data = load_dataset("imdb", split="test")
test_data = data.shard(num_shards=4, index=0)

model = AutoModelForAudioClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#tokenize the data
# tokenized_training_data = tokenizer(train_data["text"], return_tensors="pt", padding=True, truncation=True, max_length=64)
# tokenizer_test_data = tokenizer(test_data["text"], return_tensors="pt", padding=True, truncation=True, max_length=64)

#tokenizer function of row/batch tokenizing
def tokenize_function(text_data):
    return tokenizer(text_data["text"], return_tensors="pt", padding= True, truncation=True, max_length=64)

tokenized_data = train_data.map(tokenize_function, batched= True)
tokenized_test_data = test_data.map(tokenize_function, batched= True)
tokenized_train_data = train_data.map(tokenize_function, batched= True)
# tokenize_by_row = train_data.map(tokenize_function, batched=False)

epochs = 10
#Training loop for fine-tuning
training_args = TrainingArguments(
    output_dir = "./finetuned",
    evaluation_strategy="epoch",
    num_train_epochs= epochs,
    learning_rate= 0.001,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01
)

trainer = Trainer(
    model = model,
    args=training_args,
    train_dataset= tokenized_train_data,
    eval_dataset= tokenized_test_data,
    tokenizer= tokenizer
)

trainer.train()

#use finetuned model
input_text = ["I'd just like to say, I love the product! Thank you!"]
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)

predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()

label_map = {0: "Low risk", 1: "High risk"}
for i, predicted_label in enumerate(predicted_labels):
    churn_label = label_map[predicted_label]
    print(f"\n Input Text {i + 1}: {input_text[i]}")
    print(f"Predicted Label: {predicted_label}")

#save model and tokenizers
model.save_pretrained("my_finetuned_files")
tokenizer.save_pretrained("my_finetuned_files")

#load a saved model
model = AutoModelForAudioClassification.from_pretrained("my_finetuned_files")
tokenizer = AutoTokenizer.from_pretrained("my_finetuned_files")

# Include an example in the input ext: one-shot
input_text = """
Text: "The dinner we had was great and the service too."
Classify the sentiment of this sentence as either positive or negative.
Example:
Text: "The food was delicious"
Sentiment: Positive
Text: "The dinner we had was great and the service too."
Sentiment:
"""

# Apply the example to the model
result = model(input_text, max_length=100)

print(result[0]["label"])

#Evaluation
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

print(accuracy.description)
print(f"The required data types for accuracy are: {accuracy.features}.")

# classification metrics

#evaluate LLM
predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()
# Compute the metrics by comparing real and predicted labels
#validate_labels = "provided test lables"
print(accuracy.compute(references=validate_labels, predictions=predicted_labels))

#perplexity
input_text_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_text_ids, max_length=20)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text: ", generated_text)
perplexity = evaluate.load("perplexity", module_type="metric")
results = perplexity.compute(predictions=generated_text, model_id = "gpt2")
print("Perplexity: ", results['mean_perplexity'])

#BLEU
input_sentence_1 = "Hola, ¿cómo estás?"
reference_1 = [
     ["Hello, how are you?", "Hi, how are you?"]
     ]
input_sentences_2 = ["Hola, ¿cómo estás?", "Estoy genial, gracias."]
references_2 = [
     ["Hello, how are you?", "Hi, how are you?"],
     ["I'm great, thanks.", "I'm great, thank you."]
     ]
bleu = evaluate.load("blue")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
translated_output = translator(input_sentence_1)
translated_sentence = translated_output[0]['translation_text']

print("Translated:", translated_sentence)

results = bleu.compute(predictions=[translated_sentence], references=reference_1)
print(results)

translated_outputs = translator(input_sentences_2)
predictions = [translated_output['translation_text'] for translated_output in translated_outputs]
print(predictions)

results = bleu.compute(predictions=predictions, references=references_2)
print(results)

#ROUGE -> similarity between generated a summary and refrence summaries
rouge = evaluate.load("rouge")
predictions = ["""Pluto is a dwarf planet in our solar system, located in the Kuiper Belt beyond Neptune, and was formerly considered the ninth planet until its reclassification in 2006."""]
references = ["""Pluto is a dwarf planet in the solar system, located in the Kuiper Belt beyond Neptune, and was previously deemed as a planet until it was reclassified in 2006."""]
# Calculate the rouge scores between the predicted and reference summaries
results = rouge.compute(predictions=predictions, references=references)
print("ROUGE results: ", results)

#METEOR
meteor = evaluate.load("meteor")
generated = ["The burrow stretched forward like a narrow corridor for a while, then plunged abruptly downward, so quickly that Alice had no chance to stop herself before she was tumbling into an extremely deep shaft."]
reference = ["The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well."]
# Compute and print the METEOR score
results = meteor.compute(predictions= generated, references= reference)
print("Meteor: ", results)

#ExactMatch(EM)
exact_match = evaluate.load("exact_match")
predictions = ["It's a wonderful day", "I love dogs", "DataCamp has great AI courses", "Sunshine and flowers"]
references = ["What a wonderful day", "I love cats", "DataCamp has great AI courses", "Sunsets and flowers"]
# Compute the exact match and print the results
results = exact_match.compute(predictions = predictions, references=references)
print("EM results: ", results)

#Safeguarding LLMs
#Toxicity

toxicity_metric = evaluate.load("toxicity")
texts_1 = ["Everyone in the team adores him", "He is a true genius, pure talent"]
texts_2 = ["Nobody in the team likes him", "He is a useless 'good-for-nothing'"]
toxicity_results_1 = toxicity_metric.compute(
    predictions=texts_1,
    aggregation="maximum")
toxicity_results_2 = toxicity_metric.compute(
    predictions=texts_2,
      aggregation="maximum")
print("Toxicity Sentences 1:", toxicity_results_1)
print("Toxicity Sentences 2:", toxicity_results_2)

# Calculate the toxicity ratios
toxicity_1_ratio = toxicity_metric.compute(predictions=user_1, aggregation="ratio")
toxicity_2_ratio = toxicity_metric.compute(predictions=user_2, aggregation="ratio")
print("Toxicity ratio (user_1):", toxicity_1_ratio['toxicity_ratio'])
print("Toxicity ratio (user_2): ", toxicity_2_ratio['toxicity_ratio'])

#regard anad regard-comparison
group1 = ['abc are described as loyal', 
          'abc are honest but kind']
group2 = ['abc are known for being confrontational',
          'abc are very blunt']

regard = evaluate.load("regard")
regard_comp = evaluate.load("regard", "compare")
polarity_results_1 = regard.compute(data=group1)
polarity_results_2 = regard.compute(data = group2)
print("Polarity in group 1:\n", polarity_results_1)
print("Polarity in group 2:\n", polarity_results_2)

polarity_results_comp = regard_comp.compute(data=group1, references=group2)
print("Polarity comparison between groups:\n", polarity_results_comp)
