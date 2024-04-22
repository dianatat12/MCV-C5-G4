# C5-Project-Week 6 Multimodal Human Analysis

We are group 4, composed of:
- Aleix Pujol - aleixpujolcv@gmail.com
- Diana Tat - dianatat120@gmail.com
- Georg Herodes - georgherodes99@gmail.com
- Gunjan Paul - gunjan.mtbpaul@gmail.com

Link to the presentation: https://docs.google.com/presentation/d/1Nq_7xpg0kE_MP_0IyT45t2iKNSOiyvHbd-jMPDyXkHY/edit?usp=sharing


## Instructions to run the code

## SETUP:

1. create virtual environment
2. run `pip install -r requirements.txt`

## TASKS:
### Task b
The main code for this task can be found in `task_b.py` with functions and classes such as the `Datamodule` and `Model` being imported from `utils/data.py` and `utils/model.py`. The script takes no command-line arguments and can be run using `python3 task_b.py`, which trains our image classifier model and evaluates it on the test set after the training is finished. `task_b_test.py` is used to generate a CSV file compatible with the provided evaluation script.

### Task f
For task f, you need to run the task_f.ipynb which can be found in the task_f folder. After running the code, a PCA visualization plot will be displayed showing the representations of Bag of Words (BoW) and BERT embeddings in two dimensions.

### Task g
The main code for this task can be found in `task_g.py` with functions and classes such as the `Datamodule` and `Model` being imported from `utils/multimodal_data.py` and `utils/multimodal_model.py`. The script takes no command-line arguments and can be run using `python3 task_g.py`, which trains our multimodal model and evaluates it on the test set after the training is finished. `task_g_test.py` is used to generate a CSV file compatible with the provided evaluation script.

