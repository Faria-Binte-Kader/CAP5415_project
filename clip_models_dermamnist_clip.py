############################################
############################################

##Models in OpenCLIP
#
# import open_clip
# print(open_clip.list_pretrained())
# #
# #Skip
# #Skip
# #Skip
# #
# [('RN50', 'openai'), ('RN50', 'yfcc15m'), ('RN50', 'cc12m'), ('RN101', 'openai'), ('RN101', 'yfcc15m'), 
# ('RN50x4', 'openai'), ('RN50x16', 'openai'), ('RN50x64', 'openai'), ('ViT-B-32', 'openai'), 
# ('ViT-B-32', 'laion400m_e31'), ('ViT-B-32', 'laion400m_e32'), ('ViT-B-32', 'laion2b_e16'), 
# ('ViT-B-32', 'laion2b_s34b_b79k'), ('ViT-B-32', 'datacomp_xl_s13b_b90k'), ('ViT-B-32', 'datacomp_m_s128m_b4k'), 
# ('ViT-B-32', 'commonpool_m_clip_s128m_b4k'), ('ViT-B-32', 'commonpool_m_laion_s128m_b4k'), ('ViT-B-32', 'commonpool_m_image_s128m_b4k'), 
# ('ViT-B-32', 'commonpool_m_text_s128m_b4k'), ('ViT-B-32', 'commonpool_m_basic_s128m_b4k'), ('ViT-B-32', 'commonpool_m_s128m_b4k'), 
# ('ViT-B-32', 'datacomp_s_s13m_b4k'), ('ViT-B-32', 'commonpool_s_clip_s13m_b4k'), ('ViT-B-32', 'commonpool_s_laion_s13m_b4k'), 
# ('ViT-B-32', 'commonpool_s_image_s13m_b4k'), ('ViT-B-32', 'commonpool_s_text_s13m_b4k'), ('ViT-B-32', 'commonpool_s_basic_s13m_b4k'), 
# ('ViT-B-32', 'commonpool_s_s13m_b4k'), ('ViT-B-32', 'metaclip_400m'), ('ViT-B-32', 'metaclip_fullcc'), ('ViT-B-32-256', 'datacomp_s34b_b86k'), 
# ('ViT-B-16', 'openai'), ('ViT-B-16', 'laion400m_e31'), ('ViT-B-16', 'laion400m_e32'), ('ViT-B-16', 'laion2b_s34b_b88k'), 
# ('ViT-B-16', 'datacomp_xl_s13b_b90k'), ('ViT-B-16', 'datacomp_l_s1b_b8k'), ('ViT-B-16', 'commonpool_l_clip_s1b_b8k'), 
# ('ViT-B-16', 'commonpool_l_laion_s1b_b8k'), ('ViT-B-16', 'commonpool_l_image_s1b_b8k'), ('ViT-B-16', 'commonpool_l_text_s1b_b8k'), 
# ('ViT-B-16', 'commonpool_l_basic_s1b_b8k'), ('ViT-B-16', 'commonpool_l_s1b_b8k'), ('ViT-B-16', 'dfn2b'), ('ViT-B-16', 'metaclip_400m'), 
# ('ViT-B-16', 'metaclip_fullcc'), ('ViT-B-16-plus-240', 'laion400m_e31'), ('ViT-B-16-plus-240', 'laion400m_e32'), ('ViT-L-14', 'openai'), 
# ('ViT-L-14', 'laion400m_e31'), ('ViT-L-14', 'laion400m_e32'), ('ViT-L-14', 'laion2b_s32b_b82k'), ('ViT-L-14', 'datacomp_xl_s13b_b90k'), 
# ('ViT-L-14', 'commonpool_xl_clip_s13b_b90k'), ('ViT-L-14', 'commonpool_xl_laion_s13b_b90k'), ('ViT-L-14', 'commonpool_xl_s13b_b90k'), 
# ('ViT-L-14', 'metaclip_400m'), ('ViT-L-14', 'metaclip_fullcc'), ('ViT-L-14', 'dfn2b'), ('ViT-L-14-336', 'openai'), ('ViT-H-14', 'laion2b_s32b_b79k'), 
# ('ViT-H-14', 'metaclip_fullcc'), ('ViT-H-14', 'dfn5b'), ('ViT-H-14-378', 'dfn5b'), ('ViT-g-14', 'laion2b_s12b_b42k'), 
# ('ViT-g-14', 'laion2b_s34b_b88k'), ('ViT-bigG-14', 'laion2b_s39b_b160k'), ('ViT-bigG-14', 'metaclip_fullcc'), 
# ('roberta-ViT-B-32', 'laion2b_s12b_b32k'), ('xlm-roberta-base-ViT-B-32', 'laion5b_s13b_b90k'), 
# ('xlm-roberta-large-ViT-H-14', 'frozen_laion5b_s13b_b90k'), ('convnext_base', 'laion400m_s13b_b51k'), 
# ('convnext_base_w', 'laion2b_s13b_b82k'), ('convnext_base_w', 'laion2b_s13b_b82k_augreg'), ('convnext_base_w', 'laion_aesthetic_s13b_b82k'), 
# ('convnext_base_w_320', 'laion_aesthetic_s13b_b82k'), ('convnext_base_w_320', 'laion_aesthetic_s13b_b82k_augreg'), 
# ('convnext_large_d', 'laion2b_s26b_b102k_augreg'), ('convnext_large_d_320', 'laion2b_s29b_b131k_ft'), 
# ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup'), ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg'), 
# ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_rewind'), ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_soup'), 
# ('coca_ViT-B-32', 'laion2b_s13b_b90k'), ('coca_ViT-B-32', 'mscoco_finetuned_laion2b_s13b_b90k'), ('coca_ViT-L-14', 'laion2b_s13b_b90k'), 
# ('coca_ViT-L-14', 'mscoco_finetuned_laion2b_s13b_b90k'), ('EVA01-g-14', 'laion400m_s11b_b41k'), ('EVA01-g-14-plus', 'merged2b_s11b_b114k'), 
# ('EVA02-B-16', 'merged2b_s8b_b131k'), ('EVA02-L-14', 'merged2b_s4b_b131k'), ('EVA02-L-14-336', 'merged2b_s6b_b61k'), 
# ('EVA02-E-14', 'laion2b_s4b_b115k'), ('EVA02-E-14-plus', 'laion2b_s9b_b144k'), ('ViT-B-16-SigLIP', 'webli'), ('ViT-B-16-SigLIP-256', 'webli'), 
# ('ViT-B-16-SigLIP-i18n-256', 'webli'), ('ViT-B-16-SigLIP-384', 'webli'), ('ViT-B-16-SigLIP-512', 'webli'), ('ViT-L-16-SigLIP-256', 'webli'), 
# ('ViT-L-16-SigLIP-384', 'webli'), 
# ('ViT-SO400M-14-SigLIP', 'webli'), ('ViT-SO400M-16-SigLIP-i18n-256', 'webli'), ('ViT-SO400M-14-SigLIP-378', 'webli'), 
# ('ViT-SO400M-14-SigLIP-384', 'webli'), 
# ('ViT-L-14-CLIPA', 'datacomp1b'), ('ViT-L-14-CLIPA-336', 'datacomp1b'), ('ViT-H-14-CLIPA', 'datacomp1b'), ('ViT-H-14-CLIPA-336', 'laion2b'), 
# ('ViT-H-14-CLIPA-336', 'datacomp1b'), ('ViT-bigG-14-CLIPA', 'datacomp1b'), ('ViT-bigG-14-CLIPA-336', 'datacomp1b'), 
# ('nllb-clip-base', 'v1'), ('nllb-clip-large', 'v1'), ('nllb-clip-base-siglip', 'v1'), ('nllb-clip-base-siglip', 'mrl'), 
# ('nllb-clip-large-siglip', 'v1'), ('nllb-clip-large-siglip', 'mrl'), 
# ('MobileCLIP-S1', 'datacompdr'), ('MobileCLIP-S2', 'datacompdr'), ('MobileCLIP-B', 'datacompdr'), ('MobileCLIP-B', 'datacompdr_lt'), 
# ('ViTamin-S', 'datacomp1b'), ('ViTamin-S-LTT', 'datacomp1b'), ('ViTamin-B', 'datacomp1b'), ('ViTamin-B-LTT', 'datacomp1b'), 
# ('ViTamin-L', 'datacomp1b'), ('ViTamin-L-256', 'datacomp1b'), ('ViTamin-L-336', 'datacomp1b'), ('ViTamin-L-384', 'datacomp1b'), 
# ('ViTamin-L2', 'datacomp1b'), ('ViTamin-L2-256', 'datacomp1b'), ('ViTamin-L2-336', 'datacomp1b'), ('ViTamin-L2-384', 'datacomp1b'), 
# ('ViTamin-XL-256', 'datacomp1b'), ('ViTamin-XL-336', 'datacomp1b'), ('ViTamin-XL-384', 'datacomp1b'), 
# ('RN50-quickgelu', 'openai'), ('RN50-quickgelu', 'yfcc15m'), ('RN50-quickgelu', 'cc12m'), 
# ('RN101-quickgelu', 'openai'), ('RN101-quickgelu', 'yfcc15m'), ('RN50x4-quickgelu', 'openai'), ('RN50x16-quickgelu', 'openai'), 
# ('RN50x64-quickgelu', 'openai'), ('ViT-B-32-quickgelu', 'openai'), ('ViT-B-32-quickgelu', 'laion400m_e31'), 
# ('ViT-B-32-quickgelu', 'laion400m_e32'), ('ViT-B-32-quickgelu', 'metaclip_400m'), ('ViT-B-32-quickgelu', 'metaclip_fullcc'), 
# ('ViT-B-16-quickgelu', 'openai'), ('ViT-B-16-quickgelu', 'dfn2b'), ('ViT-B-16-quickgelu', 'metaclip_400m'), ('ViT-B-16-quickgelu', 'metaclip_fullcc'), 
# ('ViT-L-14-quickgelu', 'openai'), ('ViT-L-14-quickgelu', 'metaclip_400m'), ('ViT-L-14-quickgelu', 'metaclip_fullcc'), ('ViT-L-14-quickgelu', 'dfn2b'), 
# ('ViT-L-14-336-quickgelu', 'openai'), ('ViT-H-14-quickgelu', 'metaclip_fullcc'), ('ViT-H-14-quickgelu', 'dfn5b'), ('ViT-H-14-378-quickgelu', 'dfn5b'), 
# ('ViT-bigG-14-quickgelu', 'metaclip_fullcc')]



############################################
############################################

#Models to use
#https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_classification_results.csv

#name	pretrained	params (M)	FLOPs (B)



############################################
############################################

#Import necessary libraries

import open_clip
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import datasets, transforms

from dataset import get_medmnist_dataloader_open_clip, DATASET_CLASSES
from tqdm import tqdm

############################################
############################################

#Get Model and Tokenizer with corresponding preprocess functions
def get_model_tokenizer_preprocessor(model_name, pretrained_weights):
    # Choose the model and pre-trained weights
    model_name = model_name #"ViT-B-32"  # Example model
    pretrained_weights = pretrained_weights #"openai"

    # Load model and tokenizer
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name, pretrained_weights
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    return model, preprocess_train, preprocess_val, tokenizer



#Clip (CLIP)
model, preprocess_train, preprocess_val, tokenizer = get_model_tokenizer_preprocessor("ViT-B-16", "datacomp_xl_s13b_b90k")







# print("Model")
# print(model)
# print("############################################")
# print("############################################")
# print("preprocess_train")
# print(preprocess_train)
# print("############################################")
# print("############################################")
# print("preprocess_val")
# print(preprocess_val)
# print("############################################")
# print("############################################")
# print("tokenizer")
# print(tokenizer)
# print("############################################")
# print("############################################")
# end


############################################
############################################





#Dataset lists
dataset_classes = [cls for cls in DATASET_CLASSES] 

#Create Train and Test Loader
train_loader, test_loader = get_medmnist_dataloader_open_clip(dataset_classes[5], preprocess_train, preprocess_val, batch_size=32, download=True)
#

#

#


############################################
############################################

from torch.optim import AdamW
from open_clip.loss import ClipLoss

#Declare device to be used and send model to it
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

##Classifier
#print(model.visual)
# Create a linear classifier on top of image encoder
#Doesn't work when directly called, Find a work around rather. Now declare with direct number
#classifier = nn.Linear(model.visual.ln_post, len(train_loader.dataset.dataset.dataset_class)).to(device)
classifier = nn.Linear(512, 7).to(device)


# Define loss and optimizer
#loss_fn = ClipLoss(model.logit_scale).to(device)
loss_fn_ce = nn.CrossEntropyLoss()
loss_fn_contra = ClipLoss().to(device)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)



############################################
############################################




############################################
############################################


from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Initialize scheduler
scheduler = MultiStepLR(optimizer, milestones=[7, 9], gamma=0.1)

# Initialize variables to track the best metrics
best_f1 = 0
best_accuracy = 0
best_model_path = "dermamnist_ViT_B_16_clip_model.pth"

# Metrics storage
metrics = {"train": [], "test": []}

for epoch in tqdm(range(10)):
    print(f"Epoch {epoch+1}/10")
    
    # Training phase
    model.train()
    train_predictions = []
    train_labels = []
    total_loss = 0

    for images, texts, labels in tqdm(train_loader):
        images = images.to(device)
        texts = tokenizer(texts).to(device)
        labels = labels.to(device)

        # Forward pass
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)

        ##Print Feature Shape so one can set linear shape manually
        # print(image_features.shape)

        #classifier
        logits_ce = classifier(image_features)
        loss_ce = loss_fn_ce(logits_ce, labels.squeeze())

        # Compute loss
        loss = 0.0001 * loss_fn_contra(image_features, text_features, model.logit_scale.exp())
        
        total_loss += loss_ce.item() + loss.item()

        loss_ce = loss_ce + loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss_ce.backward()
        optimizer.step()

        # Collect predictions and labels for metrics
        predictions = logits_ce.argmax(dim=1).cpu().numpy()
        train_predictions.extend(predictions)
        train_labels.extend(labels.cpu().numpy())        


        # #
        # print(train_predictions)
        # print(train_labels)

    # Calculate training metrics
    train_accuracy = accuracy_score(train_labels, train_predictions)
    train_precision = precision_score(train_labels, train_predictions, average="weighted")
    train_recall = recall_score(train_labels, train_predictions, average="weighted")
    train_f1 = f1_score(train_labels, train_predictions, average="weighted")
    metrics["train"].append(
        {"accuracy": train_accuracy, "precision": train_precision, "recall": train_recall, "f1": train_f1}
    )

    print(f"Training Loss: {total_loss / len(train_loader):.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")

    # Validation phase
    model.eval()
    test_predictions = []
    test_labels = []
    val_loss = 0

    with torch.no_grad():
        for images, texts, labels in tqdm(test_loader):
            images = images.to(device)
            texts = tokenizer(texts).to(device)
            labels = labels.to(device)

            # Forward pass
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            ##Print Feature Shape so one can set linear shape manually
            # print(image_features.shape)

            #classifier
            logits_ce = classifier(image_features)
            val_loss_ce = loss_fn_ce(logits_ce, labels.squeeze())

            # Compute loss
            val_loss_fn = 0.0001 * loss_fn_contra(image_features, text_features, model.logit_scale.exp())

            val_loss += val_loss_ce.item() + val_loss_fn.item()


            predictions = logits_ce.argmax(dim=1).cpu().numpy()
            test_predictions.extend(predictions)
            test_labels.extend(labels.cpu().numpy())    



    # Calculate testing metrics
    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_precision = precision_score(test_labels, test_predictions, average="weighted")
    test_recall = recall_score(test_labels, test_predictions, average="weighted")
    test_f1 = f1_score(test_labels, test_predictions, average="weighted")
    metrics["test"].append(
        {"accuracy": test_accuracy, "precision": test_precision, "recall": test_recall, "f1": test_f1}
    )

    print(f"Validation Loss: {val_loss / len(test_loader):.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")

    # Save the best model
    if test_f1 > best_f1 or (test_f1 == best_f1 and test_accuracy > best_accuracy):
        best_f1 = test_f1
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with F1: {best_f1:.4f}, Accuracy: {best_accuracy:.4f}")

    # Step the scheduler
    scheduler.step()

# Save metrics to a file
import json
with open("dermamnist_ViT_B_16_clip_metrics.json", "w") as f:
    json.dump(metrics, f)
print("Training complete. Best model saved as:", best_model_path)



############################################
############################################





############################################
############################################




############################################
############################################






############################################
############################################





############################################
############################################




############################################
############################################






############################################
############################################





############################################
############################################

