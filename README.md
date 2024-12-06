# CAP5415_project
# Instructions
If your are using any IDE then first clone (git clone <url>) the repository. Then create a virtual environment and activate it. <br>

`conda env create -f environment.yml`<br>
`conda activate op_clip`

For zero-shot clasification on any subset of MedMNIST dataset-
```
python zero_shot.py \
  --dataset_class BreastMNIST \
  --batch_size 32 \
  --model_name openai/clip-vit-base-patch16 \
  --model_type clip
```
You can specify any subset of MedMNIST (BloodMNIST, PathMNIST, OCTMNIST, PneumoniaMNIST, DermaMNIST, RetinaMNIST, BreastMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST, TissueMNIST) here.

To train our framework model with any of the subsets, you can simply run the python file with the corresponding dataset name-
```
python clip_models_bloodmnist_clip.py
```
This will also generate the corresponding accuracy and F1-scores.


For Fine-Tuned CLIP with custom loss function on the subsets of MedMNIST dataset, run the following,
```
python clip_models_bloodmnist_clip.py
python clip_models_breastmnist_clip.py
python clip_models_chestmnist_clip.py
python clip_models_dermamnist_clip.py
python clip_models_octmnist_clip
python clip_models_organAmnist_clip.py
python clip_models_organCmnist_clip
python clip_models_organSmnist_clip.py
python clip_models_pathmnist_clip.py
python clip_models_pneumoniamnist_clip.py
python clip_models_retinamnist_clip.py
python clip_models_tissuemnist_clip.py

```


![Model Image](“master/cs5415_vit_text.png”)
