import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import labelbox
import json
from PIL import Image, ImageFilter  
import pytesseract
import requests
import os
from sklearn.preprocessing import LabelEncoder
import re
import random
from torchvision.transforms import functional as F

label_encoder = LabelEncoder()

def create_folder(image_path):
    CHECK_FOLDER = os.path.isdir(image_path)
    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(image_path)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

dir_path = os.path.dirname(os.path.realpath(__file__))
image_path = os.path.join(dir_path, "images")
create_folder(image_path)

client = labelbox.Client(api_key='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbHQ2MmdoaGowMnlsMDd2Y2VxMHY2Ymh6Iiwib3JnYW5pemF0aW9uSWQiOiJjbHQ2MmdoaGEwMnlrMDd2Yzd1NGViaDl5IiwiYXBpS2V5SWQiOiJjbHR3OGtpbmQxMTd5MDcwcTFwMTMzamsyIiwic2VjcmV0IjoiY2RiYjlmNDA2NzFkMDljOTZlYzg0YTc5N2U3M2ExNDkiLCJpYXQiOjE3MTA3MjM0NDUsImV4cCI6MjM0MTg3NTQ0NX0.ksu2f4xC8RILe45TZoQ7nTsIKEPy8_n3S1QPNb0d47M')
params = {
    "data_row_details": False,
    "metadata_fields": False,
    "attachments": True,
    "project_details": False,
    "performance_details": False,
    "label_details": True,
    "interpolated_frames": False
}

def download_and_save_image(image_url, image_name, image_dir):
    image_path = os.path.join(image_dir, image_name)
    if not os.path.exists(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path))
    if not os.path.exists(image_path):
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(image_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"Failed to download image from {image_url}")
            return False
    return True

# Modify preprocess_data function to apply augmentation
def preprocess_data(data, image_dir):
    texts = []
    labels = []
    global label_encoder

    for project_data in data:
        for project_id, project_info in project_data['projects'].items():
            for label_data in project_info['labels']:
                for classification in label_data['annotations']['classifications']:
                    if 'radio_answer' in classification:
                        radio_answer = classification['radio_answer']
                        if radio_answer['name'] not in ['Bad Quality (so cant use)', 'Not Related']:
                            image_url = project_data['data_row']['row_data']
                            image_path = f"{project_data['data_row']['external_id']}"
                            if download_and_save_image(image_url, image_path, image_dir):
                                text_from_image = extract_text_from_image(os.path.join(image_dir, image_path))
                                # Data augmentation
                                image = Image.open(os.path.join(image_dir, image_path))
                                texts.append(text_from_image)
                                labels.append(radio_answer['name'])

    encoded_labels = label_encoder.fit_transform(labels)
    return texts, encoded_labels

def extract_text_from_image(image_path):
    try:
        # Open the image file
        image = Image.open(image_path)

        # Perform OCR
        text = pytesseract.image_to_string(image)

        return text
    except OSError as e:
        print(f"OSError: image file '{image_path}' is truncated or corrupted.")
        return None
    except Exception as e:
        print(f"Error processing image '{image_path}': {e}")
        return None


# Get the exported JSON data
export_task = client.get_project('clt63ic3506us07wvgvx2bqfr').export_v2(params=params)
export_task.wait_till_done()

# Check for errors
if export_task.errors:
    print(export_task.errors)
else:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=7)  # Assuming binary classification
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    export_json = export_task.result

    # Preprocess the data
    texts, labels = preprocess_data(export_json, image_path)

    # Split the data into training, validation, and testing sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.1, random_state=42)

    # Tokenize the texts and convert to input tensors
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']),
                                torch.tensor(train_encodings['attention_mask']),
                                torch.tensor(train_labels))
    val_dataset = TensorDataset(torch.tensor(val_encodings['input_ids']),
                                torch.tensor(val_encodings['attention_mask']),
                                torch.tensor(val_labels))
    test_dataset = TensorDataset(torch.tensor(test_encodings['input_ids']),
                                torch.tensor(test_encodings['attention_mask']),
                                torch.tensor(test_labels))

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Fine-tune BERT on the training set
    optimizer = AdamW(model.parameters(), lr=3e-5)  # Adjust learning rate
    model.train()
    num_epochs = 22  # Increase the number of epochs
    for epoch in range(num_epochs):
        for batch in train_loader:
            input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    
    # Decode the numerical labels back to their original names
    decoded_val_labels = label_encoder.inverse_transform(val_labels)
    decoded_test_labels = label_encoder.inverse_transform(test_labels)


    # Evaluate the model on the validation set
    model.eval()
    val_preds, val_labels = [], []
    for batch in val_loader:
        input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        val_preds.extend(preds)
        val_labels.extend(labels.cpu().numpy())

    # Calculate evaluation metrics on the validation set
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_precision = precision_score(val_labels, val_preds, average=None, zero_division=0)
    val_recall = recall_score(val_labels, val_preds, average=None, zero_division=0)
    val_f1 = f1_score(val_labels, val_preds, average=None, zero_division=0)

    # Print metrics per class for validation set
    print("\nValidation Metrics per Class:")
    print(f"Total Accuracy: {val_accuracy}")
    for label, precision, recall, f1 in zip(decoded_val_labels, val_precision, val_recall, val_f1):
        print(f"Class {label}: Precision={precision}, Recall={recall}, F1-score={f1}")

    # Calculate and print overall metrics for validation set
    val_total_precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)
    val_total_recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)
    val_total_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

    print("\nTotal Validation Metrics:")
    print(f"Total Precision: {val_total_precision}")
    print(f"Total Recall: {val_total_recall}")
    print(f"Total F1-score: {val_total_f1}")

    # Evaluate the model on the test set
    test_preds, test_labels = [], []
    for batch in test_loader:
        input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        test_preds.extend(preds)
        test_labels.extend(labels.cpu().numpy())

    # Calculate evaluation metrics on the test set
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds, average=None, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, average=None, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, average=None, zero_division=0)

    # Print metrics per class for test set
    print("\nTest Metrics per Class:")
    print(f"Total Accuracy: {test_accuracy}")
    for label, precision, recall, f1 in zip(decoded_test_labels, test_precision, test_recall, test_f1):
        print(f"Class {label}: Precision={precision}, Recall={recall}, F1-score={f1}")

    # Calculate and print overall metrics for test set
    test_total_precision = precision_score(test_labels, test_preds, average='macro', zero_division=0)
    test_total_recall = recall_score(test_labels, test_preds, average='macro', zero_division=0)
    test_total_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)

    print("\nTotal Test Metrics:")
    print(f"Total Precision: {test_total_precision}")
    print(f"Total Recall: {test_total_recall}")
    print(f"Total F1-score: {test_total_f1}")
