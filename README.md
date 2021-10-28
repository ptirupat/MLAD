# MLAD
Implementation of paper "Modeling Multi-Label Action Dependencies for Temporal Action Localization"

https://openaccess.thecvf.com/content/CVPR2021/papers/Tirupattur_Modeling_Multi-Label_Action_Dependencies_for_Temporal_Action_Localization_CVPR_2021_paper.pdf

Here is the sample command to train the model on MultiTHUMOS dataset 

python3 main.py --train_classifier --gpu 0 --run_id multithumos_v1_5layers --run_description " Experiment with 5 Transformer Encoder layers with varied length input." --dataset multithumos --model_version 'v1' --train_mode 'fixed' --eval_mode 'slide' --input_type "combined" --num_clips 128 --skip 0 --feature_dim 2048 --hidden_dim 128 --num_layers 5 --batch_size 32 --num_epochs 2500 --num_workers 0 --learning_rate 1e-3 --weight_decay 1e-6 --optimizer ADAM --f1_threshold 0.5 --varied_length


### Trained Models
Here are the trained models

Charades : https://drive.google.com/file/d/1tna5PLkFm2A9RA45sOtCnG6yx6mGHw4j/view?usp=sharing

MultiTHUMOS : https://drive.google.com/file/d/1vXq-y68hC4Qe6N1PBk3DlqGjOWhP9Vsc/view?usp=sharing

These are the models with 5 MLAD layers and give the best results on the corresponding datasets.

### Features
Download the features used for training the models from the following links

MultiTHUMOS : https://drive.google.com/drive/folders/1txv4OyMd88ku3nzWAeYVhJ-9YR8NHE8w?usp=sharing


### TODO
* Add code to visualize the results 
