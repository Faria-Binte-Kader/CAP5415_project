# CAP5415_project
# Instructions
If your are using any IDE then first clone (git clone <url>) the repository. Then create a virtual environment and activate it. <br>

`conda create -n CAP5415 Python=3.10.12`<br>
`conda activate CAP5415`

Install all the dependencies.<br>
`pip install -r requirements.txt`

For zero-shot clasification on any subset of MedMNIST dataset-
```
python zero_shot.py \
  --dataset_class BreastMNIST \
  --batch_size 32 \
  --model_name openai/clip-vit-base-patch16 \
  --model_type clip
```
You can specify any subset of MedMNIST (BloodMNIST, PathMNIST, OCTMNIST, PneumoniaMNIST, DermaMNIST, RetinaMNIST, BreastMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST, TissueMNIST) here.
