# Natural Language Processing - Medical Diagnosis

This project is a solution to help on medical diagnosis, using natural language processing.
Two models (Naive Bayes Classifier and CNN) were developed to compare performances and both were saved in Models/Saved.
The best performance was obtained with Naive Bayes Classifier, therefore that is the model incorporated with the application running.

### Installation

Run the following command to install the needed python libraries prior to run the code

```angular2html
pip install -r requirements.txt
```

### How to run the code

This project can be used to perform the training + testing phase and application phase, or just the application phase.
To run with training, testing and application phases please run:

```angular2html
python main.py train
```

To run only the application use:

```
python main.py
```