import argparse
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import classification_report
from dataset import get_medmnist_dataloader, DATASET_CLASSES

# Helper function to load the appropriate model and processor
def load_model_and_processor(model_name, model_type):
    if model_type.lower() == "clip":
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model, processor

# Zero-shot classification function
def zero_shot_classification(dataset_class, batch_size, model_name, model_type):
    # Load the model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model_and_processor(model_name, model_type)
    model = model.to(device)
    
    # Create a test dataloader
    _, test_loader = get_medmnist_dataloader(dataset_class, batch_size=batch_size)

    # Evaluation loop
    model.eval()
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for images, captions, labels in test_loader:
            # Move inputs to device
            images = images.to(device)
            labels = labels.to(device)

            # Preprocess text captions and images
            text_inputs = processor(text=captions, return_tensors="pt", padding=True).to(device)
            image_inputs = processor(images=images, return_tensors="pt").to(device)

            # Perform forward pass
            outputs = model(input_ids=text_inputs['input_ids'], attention_mask=text_inputs['attention_mask'],
                            pixel_values=image_inputs['pixel_values'])

            # Get logits for classification
            logits_per_image = outputs.logits_per_image  # Shape: [batch_size, num_classes]

            
            # Get predictions
            predictions = torch.argmax(logits_per_image, dim=1)
            # Store predictions and ground truth
            all_predictions.extend(predictions.cpu().tolist())
            all_ground_truths.extend(labels.cpu().tolist())

    # Evaluate metrics
    report = classification_report(all_ground_truths, all_predictions, output_dict=True, zero_division=0)
    accuracy = report["accuracy"]
    precision = report["macro avg"]["precision"]
    recall = report["macro avg"]["recall"]
    f1_score = report["macro avg"]["f1-score"]

    # Print metrics
    print(dataset_class)
    print(f"Zero-shot classification accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1_score * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot classification using CLIP, CoCa, or SigLIP.")
    parser.add_argument("--dataset_class", type=str, required=True, help="Name of the MedMNIST dataset class.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch16", help="Name of the model.")
    parser.add_argument("--model_type", type=str, choices=["clip"], required=True, help="Type of the model to use.")
    
    args = parser.parse_args()
    
    # Find the dataset class object by name
    dataset_class = next((cls for cls in DATASET_CLASSES if cls.__name__ == args.dataset_class), None)
    if dataset_class is None:
        raise ValueError(f"Dataset class {args.dataset_class} not found in MedMNIST.")
    
    # Run the zero-shot classification
    zero_shot_classification(dataset_class, args.batch_size, args.model_name, args.model_type)
