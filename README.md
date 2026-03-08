Human Activity Recognition using Hidden Markov Models
A machine learning project that uses Hidden Markov Models (HMMs) to classify human activities from smartphone sensor data (accelerometer and gyroscope).

Project Overview
This project collects real motion data using the Sensor Logger app, extracts meaningful features from the raw signals, and trains a Hidden Markov Model to infer which activity a person is performing at any given moment.
The four activities recognised are:

Jumping — continuous jumps recorded at waist level
Walking — consistent pace, phone at waist level
Standing — phone held steady, minimal movement
Motionless — phone placed flat on a surface


Repository Structure
├── dataset_for_hmm/          # Training dataset (collected by Member 1)
│   ├── jumping/
│   │   ├── rec-01/
│   │   │   ├── Accelerometer.csv
│   │   │   └── Gyroscope.csv
│   │   ├── rec-02/ ...
│   │   └── rec-06/
│   ├── motionless/           # rec-01 to rec-06
│   ├── standing/             # rec-01 to rec-07
│   └── walking/              # rec-01 to rec-06
│
├── dataset/                  # Test dataset (collected by Member 2)
│   ├── jumping/
│   ├── motionless/
│   ├── standing/
│   └── walking/
│
├── notebook
|   |__ 
└── README.md                 # This file

Data Collection
DetailValueApp usedSensor Logger (iOS/Android)Sensors recordedAccelerometer (x, y, z), Gyroscope (x, y, z)Sampling rate~99 HzTotal recordings25 (training) + 25 (testing)Duration per recording5–10 secondsFile formatCSV with time, seconds_elapsed, x, y, z columns
Data Split Strategy

Training data (dataset_for_hmm): all 25 recordings collected by Member 1
Test data (dataset): all 25 recordings collected by Member 2 in a separate session

This ensures the model is evaluated on completely unseen data from a different participant, providing a true measure of generalisation.

Methodology
1. Feature Extraction
Raw sensor readings are sliced into 1-second sliding windows (100 samples) with 50% overlap. Each window is compressed into 27 features:
Feature TypeFeaturesAxesTime-domainMean, Standard DeviationAll 6 axesFrequency-domainDominant Frequency, Spectral EnergyAll 6 axesCombinedSignal Magnitude Area (SMA), Correlation (XY, XZ)Accelerometer
2. Model Architecture
One Gaussian HMM is trained per activity (4 models total), each with:

3 hidden states — to capture sub-phases within each activity (e.g. jump launch, airborne, landing)
Diagonal covariance — for numerical stability with limited data
Baum-Welch algorithm — to learn transition and emission probabilities from training data

3. Prediction
To classify a new sequence, it is scored against all 4 models. The model returning the highest log-likelihood determines the predicted activity.
4. Decoding
The Viterbi algorithm is used to decode the most likely sequence of hidden states for a given observation sequence, revealing the internal sub-phases of each activity over time.

Results
Overall Accuracy on unseen test dataset: 85.08%
ActivitySamplesSensitivitySpecificityAccuracyJumping112100.00%88.01%91.14%Motionless99100.00%100.00%100.00%Standing129100.00%91.33%93.94%Walking8928.09%100.00%85.08%
Key Observations

Motionless achieved perfect classification — the near-flat signal is highly distinctive
Jumping and Standing were identified with 100% sensitivity
Walking showed low sensitivity (28%) due to inter-person variability — the model was trained on one person's walking pattern and tested on another person's, causing misclassification. This is a known limitation of activity recognition models trained on single-participant data.


How to Run
Requirements
hmmlearn
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
Steps (Google Colab)

Upload dataset_for_hmm.zip, dataset.zip, and human_activity_hmm.ipynb to Colab
Open the notebook and uncomment the unzip lines in Cell 2:

python!unzip dataset_for_hmm.zip -d .
!unzip dataset.zip -d .

Run all cells in order from top to bottom

Steps (Local)

Clone or download this repository
Install dependencies:

bashpip install hmmlearn pandas numpy scipy scikit-learn matplotlib seaborn

Ensure both dataset folders are in the same directory as the notebook
Open and run human_activity_hmm.ipynb


Notebook Structure
CellDescription1Install and import all libraries2Set dataset paths and configuration constants3Load both datasets into DataFrames4Visualise raw accelerometer signals per activity5Extract 27 features using sliding windows6Scale features using StandardScaler7Train one Gaussian HMM per activity (Baum-Welch)8Visualise learned transition matrices9Predict on unseen test data10Plot confusion matrix11Viterbi decode and plot hidden state sequences12Compute sensitivity, specificity, and accuracy per activity

Limitations & Future Improvements

Inter-person variability: training and testing on different participants reduces walking accuracy. Including data from multiple people during training would improve generalisation.
Dataset size: with only ~25 recordings per person, the model has limited exposure to activity variation. More recordings would improve robustness.
Additional features: adding features like jerk (rate of change of acceleration) or energy in specific frequency bands could help distinguish walking from standing more reliably.
Additional sensors: heart rate or barometer data could further improve classification.