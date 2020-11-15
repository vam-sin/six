# Six

Deep Learning based tool for predicting m6A sites in RNA sequences

# Requirements

```python3
pip3 install keras
pip3 install tensorflow
```

# DeepM6ASeq Results

To obtain the results of the DeepM6ASeq model on the miSeq dataset, run the following code.

```python3
python3 deepm6aseq_metrics.py
```

# Six Results

To recreate the results of the six model, please run the following commands in the mentioned step by the step fashion.

## Feature Generation

To generate the sequence motif and the secondary structure features, run the following code.

```python3
python3 feature_generation.py
```

## Model Design

To check the model summary, run the following command.

```python3
python3 model_nn.py
```

## Model Training

To train the model and obtain results on the test dataset, run the following command.

```python3
python3 train_nn.py
```
