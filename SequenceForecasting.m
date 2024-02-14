clc; clear; close all;

load WaveformData

numChannels = size(data{1}, 1);

%% Load data
figure
tiledlayout(2, 2)

for i = 1:4
    nexttile
    stackedplot(data{i}')

    xlabel("Time Step")
end

numObservations = numel(data);
idxTrain = 1:floor(0.9 * numObservations);
idxTest = (floor(0.9 * numObservations) + 1):numObservations;
dataTrain = data(idxTrain);
dataTest = data(idxTest);

%% Prepare Data for Training
for n = 1:numel(dataTrain)
    X = dataTrain{n};
    XTrain{n} = X(:, 1:end - 1);
    TTrain{n} = X(:, 2:end);
end

%% Normalize Data
muX = mean(cat(2, XTrain{:}), 2);
sigmaX = std(cat(2, XTrain{:}), 0, 2);

muT = mean(cat(2, TTrain{:}), 2);
sigmaT = std(cat(2, TTrain{:}), 0, 2);

for n = 1:numel(XTrain)
    XTrain{n} = (XTrain{n} - muX) ./ sigmaX;
    TTrain{n} = (TTrain{n} - muT) ./ sigmaT;
end

%% Define LSTM Neural Network Architecture
layers = [
          sequenceInputLayer(numChannels) % input size should match the number of channels
          lstmLayer(128) % 128 hidden units
          fullyConnectedLayer(numChannels)
          regressionLayer];

options = trainingOptions("adam", ...
    MaxEpochs = 200, ...
    SequencePaddingDirection = "left", ...
    Shuffle = "every-epoch", ...
    Plots = "training-progress", ...
    Verbose = 0);

%% Train Recurrent Neural Network
net = trainNetwork(XTrain, TTrain, layers, options);

%% Test Recurrent Neural Network
for n = 1:size(dataTest, 1)
    X = dataTest{n};
    XTest{n} = (X(:, 1:end - 1) - muX) ./ sigmaX;
    TTest{n} = (X(:, 2:end) - muT) ./ sigmaT;
end

YTest = predict(net, XTest, SequencePaddingDirection = "left");

for i = 1:size(YTest, 1)
    rmse(i) = sqrt(mean((YTest{i} - TTest{i}) .^ 2, "all"));
end

figure
histogram(rmse)
xlabel("RMSE")
ylabel("Frequency")

mean(rmse)

%% Forecast Future Time Steps
idx = 2;
X = XTest{idx};
T = TTest{idx};

figure
stackedplot(X', DisplayLabels = "Channel " + (1:numChannels))
xlabel("Time Step")
title("Test Observation " + idx)

%% Open Loop Forecasting
net = resetState(net);
offset = 75;
[net, ~] = predictAndUpdateState(net, X(:, 1:offset));

numTimeSteps = size(X, 2); % number of time steps in the test data
numPredictionTimeSteps = numTimeSteps - offset; % number of time steps to predict
Y = zeros(numChannels, numPredictionTimeSteps); % initialize the predicted output

for t = 1:numPredictionTimeSteps
    Xt = X(:, offset + t);
    [net, Y(:, t)] = predictAndUpdateState(net, Xt);
end

figure
t = tiledlayout(numChannels, 1);
title(t, "Open Loop Forecasting")

for i = 1:numChannels
    nexttile
    plot(T(i, :))
    hold on
    plot(offset:numTimeSteps, [T(i, offset) Y(i, :)], '--')
    ylabel("Channel " + i)
end

xlabel("Time Step")
nexttile(1)
legend(["Input" "Forecasted"])

%% Closed Loop Forecasting
net = resetState(net);
offset = size(X, 2);
[net, Z] = predictAndUpdateState(net, X);

numPredictionTimeSteps = 200;
Xt = Z(:, end);
Y = zeros(numChannels, numPredictionTimeSteps);

for t = 1:numPredictionTimeSteps
    [net, Y(:, t)] = predictAndUpdateState(net, Xt);
    Xt = Y(:, t);
end

numTimeSteps = offset + numPredictionTimeSteps;

figure
t = tiledlayout(numChannels, 1);
title(t, "Closed Loop Forecasting")

for i = 1:numChannels
    nexttile
    plot(T(i, 1:offset))
    hold on
    plot(offset:numTimeSteps, [T(i, offset) Y(i, :)], '--')
    ylabel("Channel " + i)
end

xlabel("Time Step")
nexttile(1)
legend(["Input" "Forecasted"])
