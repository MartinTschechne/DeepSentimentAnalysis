# Project

<p align="center"><img src="./figures/example_wav.png?" alt="sound wave" height="100" width="800"></p>

This project is about applying deep learning and advanced signal processing for sentiment analysis. It is meant as proof of concept to investigate a different approach in handling time series data.

The goal of this project is to develop a model which is able to **classify samples of emotional speech**.

# Data

For this project we use the Emo-DB data. The database is containing samples of emotional speech in German. It contains samples labeled with one of 7 different emotions: Anger, Boredom, Disgust, Fear, Happiness, Sadness and Neutral.

The database has 535 sound samples, each one spoken by one of 10 different professional voice actors (5 female, 5 male) in one of 7 emotions. The texts spoken by the actors have a neutral content, are randomly selected and do not contain any hints to the emotion, i.e. only the way a sentence is spoken gives information about the emotion not the content.

Some summary statistics about the data:
<p align="center"><img src="./figures/data_info.png?" alt="summary statistics" height="350"></p>

The Emo-DB dataset and more details about it can be found [here](http://emodb.bilderbar.info/index-1280.html).

# Model

## Train-Test-Split

A traditional train-test-split where one puts 10% of randomly selected samples aside for validation, further 10% for testing and train on the remaining 80% is not advisable here. Since there are multiple samples spoken by the same voice actor (therefore the samples are correlated), a traditional train-test-split would mean that there exists a high probability of samples spoken by the same actor being in the train _and_ the test set. This is called __Data Leakage__ and must be prevented. To achieve a model that generalizes well to new speakers it never heard before, it is important to evaluate the model on audio samples from new speakers. Therefore we split the audio samples based on their voice actors rather than by individual samples.

Further to have a final model that __does not discriminate against gender__, it is vital that we make sure to have the same amount of female speakers and  male speakers in the validation and test sets.

## Advanced Signal Processing

There are multiple ways to deal with time series data like audio signals, one of the most frequently used ones are recurrent neural network architectures like LSTMs. However this is a small proof of concept  project where main goal is to try  a different approach, build it quickly and potentially reduce the amount of computation needed which comes along with LSTM cells.

In this project we use advanced signal processing to convert the 1D audio signals into 2D [Mel-spectrograms](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum). One can think of these image-like representations as a time distributed frequency spectrum of the signal. For each time step in the signal we get a spectrum.

Example Mel-spectrogram:
<p align="center"><img src="./figures/mel_spec.png?" alt="mel spectrogram" height="200" ></p>

The 1D time series is now converted to a 2D image representation for which we can use simple computer vision methods like convolutional neural networks (CNN). __The idea is to identify unique patterns in the frequency domain which are independent of the speaker and correspond to a certain emotion.__

## CNN & Training

The model is a small CNN with two convolution layers, two dense layers and a dropout layer:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 128, 281, 1)]     0         
_________________________________________________________________
conv2d (Conv2D)              (None, 128, 281, 32)      544       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 32, 70, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 70, 64)        32832     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 17, 64)         0         
_________________________________________________________________
flatten (Flatten)            (None, 8704)              0         
_________________________________________________________________
dense (Dense)                (None, 128)               1114240   
_________________________________________________________________
dropout (Dropout)            (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 7)                 903       
=================================================================
Total params: 1,148,519,
Trainable params: 1,148,519,
Non-trainable params: 0,
_________________________________________________________________
```

To prevent overfitting the model is only trained for 10 epochs where the weights from the epoch with minimum validation loss are saved. Moreover, to cope with the limited amount of training data, we augmented the training set by random shifts in time, squeezing / stretching the signal, adding noise and flipping it. We used Adam optimizer with default learning rate and batch size of 32. __The hyperparameters are not optimized at all__, no thorough parameter search has been conducted. Since this is a proof of concept, it was more important to show that it can work rather than achieving SOTA performance.

# Evaluation

__Disclaimer__: Because this project is a simple fun proof of concept  project of no academic value the evaluation is based on a single train-test-split. A more thorough performance evaluation would be to conduct a cross-validation with every possible combination of speakers in the train and test set.

|Overall Performance | Confusion Matrix |
|---|---|
|<p align="center"><img src="./figures/precision_recall_f1.png?" alt="metrics" height="350"></p>|<p align="center"><img src="./figures/confusion_matrix.png?" alt="confusion matrix" height="350"></p>

The final evaluation shows that we achieve an accuracy of 0.71 and a F1-value of 0.71 on the hold-out test set. If we weight the predictions by their frequency we get slightly better results.  
The confusion matrix shows that the model has the most difficulties with the emotions happiness (F), neutral (N) and fear (A). Interestingly we observe that there is a lot confusion between neutral (N) and boredom (L), which makes sense since a bored voice can be easily confused with a neutral voice. Moreover happiness (F) and anger (W) being confused sometimes. A possible explanation for this is that these two emotions express themselves in a louder than normal and faster paced voice.  

A quick comparison at the performance for the female and the male speaker in the test set will show if the model has difficulties predicting the emotions for a specific gender:

<p align="center"><img src="./figures/superficial_results.png?" alt="metrics" height="200"></p>

As we see the accuracies for the female and male speaker are about the same in this case.  

Let's do a quick binomial test to check the p-value and confidence interval in R:  
`H_0`: model is randomly guessing `p = 1/7`  
`H_A`: model is not randomly guessing `p != 1/7`  
Using the results of the test set, 76 samples correct out of 107 samples total:

```r
binom.test(76,107,1/7,alternative='two.sided')

	Exact binomial test

data:  76 and 107
number of successes = 76, number of trials = 107, p-value < 2.2e-16
alternative hypothesis: true probability of success is not equal to 0.1428571
95 percent confidence interval:
 0.6146337 0.7939247
sample estimates:
probability of success
             0.7102804
```

Finally, this quick evaluation shows that
- the model has an overall accuracy of 71% on the test set,
- is better than random guessing emotions and
- does not discriminate against gender, the test scores for female and male speakers are about the same.
