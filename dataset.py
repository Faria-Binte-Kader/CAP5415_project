import torch
from torch.utils.data import DataLoader
import medmnist
from torchvision import transforms

# List of all 2D dataset classes in MedMNIST
DATASET_CLASSES = [
    medmnist.BloodMNIST,
    medmnist.PathMNIST,
    medmnist.OCTMNIST,
    medmnist.PneumoniaMNIST,
    medmnist.ChestMNIST,
    medmnist.DermaMNIST,
    medmnist.RetinaMNIST,
    medmnist.BreastMNIST,
    medmnist.OrganAMNIST,
    medmnist.OrganCMNIST,
    medmnist.OrganSMNIST,
    medmnist.TissueMNIST
]

# Function to generate label maps for each dataset (example provided for BloodMNIST)
def get_label_map(dataset_class):
    if dataset_class == medmnist.BloodMNIST:
        return {
            0: "basophil",
            1: "eosinophil",
            2: "erythroblast",
            3: "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
            4: "lymphocyte",
            5: "monocyte",
            6: "neutrophil",
            7: "platelet"
        }
    elif dataset_class == medmnist.PathMNIST:
        return {
            0: "adipose",
            1: "background",
            2: "debris",
            3: "lymphocytes",
            4: "mucus",
            5: "smooth muscle",
            6: "normal colon mucosa",
            7: "cancer-associated stroma",
            8: "colorectal adenocarcinoma epithelium"
        }
    elif dataset_class == medmnist.ChestMNIST:
        return {
            0: "atelectasis",
            1: "cardiomegaly",
            2: "effusion",
            3: "infiltration",
            4: "mass",
            5: "nodule",
            6: "pneumonia",
            7: "pneumothorax",
            8: "consolidation",
            9: "edema",
            10: "emphysema",
            11: "fibrosis",
            12: "pleural",
            13: "hernia"
        }
    elif dataset_class == medmnist.DermaMNIST:
        return {
            0: "actinic keratoses and intraepithelial carcinoma",
            1: "basal cell carcinoma",
            2: "benign keratosis-like lesions",
            3: "dermatofibroma",
            4: "melanoma",
            5: "melanocytic nev",
            6: "vascular lesions"
        }
    elif dataset_class == medmnist.OCTMNIST:
        return {
            0: "choroidal neovascularization",
            1: "diabetic macular edema",
            2: "benign keratosis-like lesions",
            3: "drusen",
            4: "normal"
        }
    elif dataset_class == medmnist.PneumoniaMNIST:
        return {
            0: "choroidal neovascularization",
            1: "diabetic macular edema",
            2: "benign keratosis-like lesions",
            3: "drusen",
            4: "normal OCT"
        }
    elif dataset_class == medmnist.RetinaMNIST:
        return {
            0: "diabetic retinopathy severity 0 level",
            1: "diabetic retinopathy severity 1 level",
            2: "diabetic retinopathy severity 2 level",
            3: "diabetic retinopathy severity 3 level",
            4: "diabetic retinopathy severity 4 level"
        }
    elif dataset_class == medmnist.BreastMNIST:
        return {
            0: "malignant",
            1: "normal, benign"
        }
    elif dataset_class == medmnist.TissueMNIST:
        return {
            0: "Collecting Duct, Connecting Tubule",
            1: "Distal Convoluted Tubule",
            2: "Glomerular endothelial cells",
            3: "Interstitial endothelial cells",
            4: "leukocytes",
            5: "Podocytes",
            6: "Proximal Tubule Segments",
            7: "Thick Ascending Limb"
        }
    elif dataset_class == medmnist.OrganAMNIST:
        return {
            0: "bladder",
            1: "femur-left",
            2: "femur-right",
            3: "heart",
            4: "kidney-left",
            5: "kidney-right",
            6: "liver",
            7: "lung-left",
            8: "lung-right",
            9: "pancreas",
            10: "spleen"
        }
    elif dataset_class == medmnist.OrganCMNIST:
        return {
            0: "bladder",
            1: "femur-left",
            2: "femur-right",
            3: "heart",
            4: "kidney-left",
            5: "kidney-right",
            6: "liver",
            7: "lung-left",
            8: "lung-right",
            9: "pancreas",
            10: "spleen"
        }
    elif dataset_class == medmnist.OrganSMNIST:
        return {
            0: "bladder",
            1: "femur-left",
            2: "femur-right",
            3: "heart",
            4: "kidney-left",
            5: "kidney-right",
            6: "liver",
            7: "lung-left",
            8: "lung-right",
            9: "pancreas",
            10: "spleen"
        }
    # Add specific label mappings for other datasets as necessary
    return {i: f"label_{i}" for i in range(10)}  # Default labels for datasets without specified names

# Define a custom dataset that modifies labels to caption format
class CaptionedMedMNIST(torch.utils.data.Dataset):
    def __init__(self, dataset_class, split='train', transform=None, download=True, size=224):
        self.dataset = dataset_class(split=split, transform=transform, download=download, size=size)
        self.label_map = get_label_map(dataset_class)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        label_caption = f"This is the picture of {self.label_map[int(label)]}"
        return image, label_caption, label

# Function to create data loaders for a specific MedMNIST dataset
def get_medmnist_dataloader(dataset_class, batch_size=32, download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])  # normalize to [-1, 1]
    ])

    train_dataset = CaptionedMedMNIST(dataset_class, split='train', transform=transform, download=download, size=224)
    test_dataset = CaptionedMedMNIST(dataset_class, split='test', transform=transform, download=download, size=224)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
