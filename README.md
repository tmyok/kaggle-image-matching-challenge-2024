# Image Matching Challenge 2024
This repository contains the code that achieved 4th place in the Image Matching Challenge 2024 competition, hosted on Kaggle.

[Image Matching Challenge 2024](https://www.kaggle.com/competitions/image-matching-challenge-2024)

Details of my solution can be found at the following link:  

[Solution Details](https://www.kaggle.com/competitions/image-matching-challenge-2024/discussion/510611)

![solution](./img/solution.drawio.svg)

# Usage

## Preparation
1. Clone this repository
   ```
   git clone https://github.com/tmyok/kaggle-image-matching-challenge-2024.git
   ```
2. Download the datasets into the input directory. For more details, refer to [input/README.md](input/README.md).
3. Run the Docker container (if necessary).
   ```
   sh docker_container.sh
   ```

## Inference

For the test data:
```
cd /kaggle/working
python3 inference.py
```

For the training data (provided by the organizer):
```
cd /kaggle/working
python3 inference.py --validation
```

Evaluation:
```
cd /kaggle/working
python3 evaluate.py
```