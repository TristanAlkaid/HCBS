import os
import re
import string
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from torch import nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device.')


class CustomModel(nn.Module):
    def __init__(self, bert_model, output_size):
        super(CustomModel, self).__init__()
        self.bert = bert_model
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.fc2 = nn.Linear(512, output_size)
        self.layer_norm = nn.LayerNorm(512)
        self.activation = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state.mean(dim=1)
        x = self.fc1(pooled_output)
        x = self.layer_norm(x)
        x = self.activation(x)
        output = self.fc2(x)
        return output.view(-1, 64, 72, 72) # [64, 72, 72]


def convert_sentence_to_tensor(sentence, tokenizer):
    tokens = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    return tokens['input_ids'], tokens['attention_mask']


def convert_dataset_to_tensors(dataset, tokenizer):
    input_tensors = []
    print("converting dataset to tensors...")
    for i in tqdm(range(len(dataset))):
        input_ids, attention_mask = convert_sentence_to_tensor(dataset[i], tokenizer)
        input_tensors.append((input_ids, attention_mask))
    return input_tensors


def train_model(model, train_data, num_epochs=1, lr=1e-3):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    total_steps = len(train_data) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        total_loss = 0
        model.train()

        for step, (input_ids, attention_mask) in enumerate(train_data, 1):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, torch.zeros_like(outputs))
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % 1000 == 0 or step == len(train_data):
                print(f'Step: {step}/{len(train_data)}, Loss: {loss.item():.4f}')

        average_loss = total_loss / len(train_data)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}')

    return model


def save_model(model, tokenizer, path):
    model_to_save = model.bert if hasattr(model, 'bert') else model
    model_to_save.save_pretrained(path)
    tokenizer.save_pretrained(path)


def load_model(path, output_size):
    model = CustomModel(BertModel.from_pretrained(path), output_size)
    return model

def test_model(model, sentence, tokenizer):
    model.to(device)
    model.eval()
    with torch.no_grad():
        input_ids, attention_mask = convert_sentence_to_tensor(sentence, tokenizer)
        output = model(input_ids.to(device), attention_mask.to(device))
    return output


def get_sentences(path="content_sentence"):
    print("load data...")
    text_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    translator = str.maketrans('', '', string.punctuation)
                    text = text.translate(translator)
                    text = re.sub(r'\n+', ' ', text)
                    text = text.lower()
                    text_list.append(text)
    # print(text_list)
    print("load data success!")
    return text_list


def get_sentence(path):
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            translator = str.maketrans('', '', string.punctuation)
            text = text.translate(translator)
            text = re.sub(r'\n+', ' ', text)
            text = text.lower()
    return text


def get_txt_files_in_folder(folder_path):
    txt_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))

    return txt_files


def save_tensor(tensor, source_path):
    list = tensor.cpu().numpy()
    file_path = source_path.replace("sentence", "numpy")
    file_path = file_path.replace(".txt", "")

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, list)


def use_model(sentence_path, model_path, model_name='bert-base-uncased'):
    print(sentence_path)
    print(f'loding {model_name} model and tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name)

    dataset = get_sentences(sentence_path)

    input_tensors = convert_dataset_to_tensors(dataset, tokenizer)

    output_size = 64 * 72 * 72
    model = CustomModel(bert_model, output_size)
    model = train_model(model, input_tensors)

    save_model(model, tokenizer, model_path)

    loaded_model = load_model(model_path, output_size)

    time_start = time.time()
    file_path_list = get_txt_files_in_folder(sentence_path)
    print("start predicting...")
    for file_path in tqdm(file_path_list):
        text = get_sentence(file_path)
        result = test_model(loaded_model, text, tokenizer)
        save_tensor(result, file_path)
    time_end = time.time()
    time_cost = time_end - time_start
    print(f"Time cost: {time_cost} s")

if __name__ == '__main__':
    use_model("/home/zxy/code/VideoLLaMA2/data/JHMDB/sentence", "./trained_model/trained_model_pooling")
