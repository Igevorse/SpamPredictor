# SMS spam predictor

This project is aimed to predict spam messages.

Multinomial Naive Bayes classifier over the "bag of words" model is used. Feature vector comprises text-term counts matrix and engineered features. Stratified 5-fold gives 0.993366 F<sub>1</sub> score.

Research, feature selection, experiments and arguments for taken decisions are in [research.ipynb](src/research.ipynb).

Production-ready solution consists of two files: [train.py](src/train.py) and [predict.py](src/predict.py).

## Install requirements

```bash
$ git clone https://github.com/Igevorse/SpamPredictor
$ cd SpamPredictor
$ virtualenv -p python3 env
$ source env/bin/activate
$ pip install -r requirements.txt

```

## Train the model

```bash
$ cd src
$ python train.py
Training...finished!
```

## Predict

```bash
$ python predict.py "Hello, how are you?"
Got sms: Hello, how are you?
Predicted class: not spam

$ python predict.py "IMPORTANT - You could be entitled up to £3,160 in compensation from mis-sold PPI on a credit card or loan. Please reply PPI for info or STOP to opt out"
Got sms: IMPORTANT - You could be entitled up to £3,160 in compensation from mis-sold PPI on a credit card or loan. Please reply PPI for info or STOP to opt out
Predicted class: spam

$ python predict.py "Well, I think I need more money. Call asap please?"
Got sms: Well, I think I need more money. Call asap please?
Predicted class: not spam

$ python predict.py "URGENT! We are trying to contact U.Todays draw shows that you have won a 2000 prize GUARANTEED. Call 090 5809 4507 from land line. Claim 3030. Valid 12hrs only"
Got sms: URGENT! We are trying to contact U.Todays draw shows that you have won a 2000 prize GUARANTEED. Call 090 5809 4507 from land line. Claim 3030. Valid 12hrs only
Predicted class: spam
``` 
