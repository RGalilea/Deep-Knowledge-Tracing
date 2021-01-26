
This repository contains my implementation of [**Deep Knowledge Tracing**](https://github.com/chrispiech/DeepKnowledgeTracing) based on the one by lccasagrande (https://github.com/lccasagrande/Deep-Knowledge-Tracing) 
for PuntajeNacional.cl


## Overview

The objective is to predict the probabilities of a student correctly answering a problem not yet seen by him.
The information needed is just what problems the student has answered, if they were correct and the dificulty of each one.

## Requirements

You'll need Python 3.7 x64 and Tensorflow 2.0 to be able to run this project. 
Tensorflow 2.0 is particularly important, because an error will pop up if you use later versions.


## Instructions

1. Clone the repository and navigate to the downloaded folder.

    ``` bash
    git clone https://github.com/RGalilea/Deep-Knowledge-Tracing.git
    cd Deep-Knowledge-Tracing
    ```

2. Install required packages:

    - If you want to install with Tensorflow-CPU, type:

        ``` bash
        pip install -e .[tf]
        ```

    - Otherwise, if you want to install with TensorFlow-GPU follow [this guide](https://www.tensorflow.org/install/) to check the necessary NVIDIA software. After that, type:

        ``` bash
        pip install -e .[tf_gpu]
        ```
    - If you need to uninstall tensorflow:
    
	``` bash
        pip uninstall tensorflow
	pip install tensorflow==2.0.0
        ```
	or 
	``` bash
        pip uninstall tensorflow
	pip install tensorflow-gpu==2.0.0
        ```


3. Navigate to the examples folder and train a network:

    - Run the python script:

        ``` bash
        python run_dkt.py -f="examples/data/[demo_dkt] Respuestas.csv" -classes="examples/data/[demo_dkt] Clasificaciones.csv" -l="nivel 1 prueba de transición"
        ```

4. Draw an influence graph for a trained network

        ``` bash
        python use_trained_DKT.py -f="examples/data/[demo_dkt] Respuestas.csv" -classes="examples/data/[demo_dkt] Clasificaciones.csv" -l="nivel 1 prueba de transición"
        ```
The folder weights/ needs to be accesible directly from were you run the file.	


## Custom Metrics

To implement custom metrics, first decode the label and then calculate the metric. This step is necessary because we encode the skill id with the label to implement the custom loss.

Here's a quick example:

```python
import tensorflow as tf
from deepkt import data_util

def bin_acc(y_true, y_pred):
    true, pred = data_util.get_target(y_true, y_pred)
    return tf.keras.metrics.binary_accuracy(true, pred, threshold=0.5)

dkt_model.compile(optimizer='rmsprop', metrics=[bin_acc])
```

Take a look at [deepkt/metrics.py](https://github.com/lccasagrande/Deep-Knowledge-Tracing/tree/master/deepkt/metrics.py) to check more examples.

## Custom Datasets

To use a different dataset, you must be sure that you have the following columns:

- **usuario_id**: The identifier of the student.
- **pregunta_id**: The identifier for the particular question.
- **correcta**: The answer to the question [0, 1].

and in the clasificaciones.csv there needs to be

- **pregunta_id**: The identifier for the particular question.
- **clasificacion_tipo**: different labels that might apply to a question. For example: Grade, dificulty, time, skill, ...
- **clasificacion**: The actual labels related to the problem. For example: 10th, easy, 20, Geometry, ...



## Support

If you have any question or find a bug, please contact me or open an issue. Pull request are also welcome.
