# ADL Final Project

## Environment
```shell=
pip install -r requirements.txt
```

## Reproduce my result
* run 
    ```shell=
    bash download.sh
    bash preprocess.sh
    ```
    to get the preprocessed data and models directly.
* Seen User Course Prediction
    ```shell=
    bash course_inference.sh seen
    ```

* Unseen User Course Prediction
    ```shell=
    bash course_inference.sh unseen
    ```
    
* Seen User Topic Prediction
    ```shell=
    bash subgroup_inference.sh seen
    ```

* Unseen User Topic Prediction
    ```shell=
    bash subgroup_inference.sh unseen
    ```

## Preprocessing
* Obtain data and place it in the team_50 folder
* To preprocess hahow dataset, run:
    ```shell=
    bash preprocess.sh
    ```


## How to train

### Course

* run:
    ```shell=
        python dropoutNet/asl.py data
    ```
    to get user matrix and item matrix
* then, run:
     ```shell=
     python dropoutNet/torch/main.py --data-dir data
     ```
    to train model

### Subgroup
* run:
    ```shell=
    python multiLabel/multi.py data
    ```
## How to test(inference)

### Course
* Seen User Course Prediction task, run:
    ```shell=
    python dropoutNet/torch/inference.py \
            --data-dir data \
            --model_path model.ckpt \
            --task seen \
            --output output.csv
    ```
* Unseen User Course Prediction task, run:
    ```shell=
    python dropoutNet/torch/inference.py \
            --data-dir data \
            --model_path model.ckpt \
            --task unseen \
            --output output.csv
    ```

### Subgroup
* Seen User Topic Prediction task, run:
    ```shell=
    python multiLabel/inference.py \
            --path data \ 
            --model_path model.ckpt \
            --task seen \
            --output output.csv
    ```
* Unseen User Topic Prediction task, run:
    ```shell=
    python multiLabel/inference.py \
            --data-dir data \
            --model_path model.ckpt \
            --task unseen \
            --output output.csv
    ```