import torch
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
