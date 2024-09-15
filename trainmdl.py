# Import necessary libraries
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
import evaluate
import numpy as np

# Define the path to your dataset
dataset_path = "dataset/"  # Replace with your actual dataset path

# Load dataset using the audiofolder loading script
dataset = load_dataset("audiofolder", data_dir=dataset_path)

# Check the current splits
print(dataset)  # This will show the existing splits (likely only "train" available)

# Manually split the dataset into train and test (80% train, 20% test)
split_dataset = dataset["train"].train_test_split(test_size=0.2)

# Combine the splits into a DatasetDict
encoded_dataset = DatasetDict({
    "train": split_dataset["train"],
    "test": split_dataset["test"]
})

# Define labels based on the folder structure
# Check if labels are strings or numerical and print them
print(encoded_dataset["train"].features["label"])  # Print to understand label structure

# Create a label mapping if labels are strings; otherwise, handle directly if numerical
label_names = encoded_dataset["train"].features["label"].names
label_to_id = {label: i for i, label in enumerate(label_names)}  # If labels are strings

# Alternatively, map numerical labels directly if applicable
if not label_to_id:
    label_to_id = {i: i for i in range(len(label_names))}  # Example mapping for numerical labels

# Print the label mapping to debug
print(f"Label Mapping: {label_to_id}")

# Load a pre-trained Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=len(label_names)
)

# Preprocess the audio data
def preprocess_function(batch):
    audio = batch["audio"]["array"]
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    batch["input_values"] = inputs.input_values[0]

    # Handling labels correctly based on their type
    if isinstance(batch["label"], int):
        # Directly use numerical labels if they are integers
        batch["labels"] = torch.tensor(batch["label"])
    elif isinstance(batch["label"], str) and batch["label"] in label_to_id:
        # Use mapped label if it's a string and present in the mapping
        batch["labels"] = torch.tensor(label_to_id[batch["label"]])
    else:
        # Raise an error with clear information if there's an unexpected label
        raise ValueError(f"Unexpected label type or missing in label_to_id: {batch['label']}")

    return batch

# Apply the preprocessing function to your dataset
encoded_dataset = encoded_dataset.map(preprocess_function)

# Load the evaluation metric using the evaluate library
metric = evaluate.load("accuracy")

# Compute metrics for evaluation
def compute_metrics(pred):
    logits = pred.predictions
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=pred.label_ids)

# Define training arguments optimized for MPS backend and lower memory usage
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Adjust this based on memory limits
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    gradient_accumulation_steps=4,  # Accumulate gradients to manage memory
    no_cuda=True,  # Ensure training on MPS or CPU
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-model")
processor.save_pretrained("./fine-tuned-model")

print("Training complete and model saved!")
