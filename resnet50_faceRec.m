clc; clear all; close all;

% Load the pre-trained model
net = resnet50();

lgraph = layerGraph(net);

% Remove last 3 layers
lgraph = removeLayers(lgraph,'fc1000');
lgraph = removeLayers(lgraph, 'fc1000_softmax');
lgraph = removeLayers(lgraph, 'ClassificationLayer_fc1000');
% New layers
tempLayers = [
    fullyConnectedLayer(6,"Name","fc")
    softmaxLayer("Name","fc1000_softmax")
    classificationLayer("Name","classoutput")];
% Adding layers
lgraph = addLayers(lgraph, tempLayers);

% Import training & validation data
imdsTrain = imageDatastore("C:\Users\Billy\Desktop\Face_Recognition_DL\face_dataset_Training","IncludeSubfolders",true,"LabelSource","foldernames");
[imdsTrain, imdsValidation] = splitEachLabel(imdsTrain,0.7,"randomized");

% Resize the images to match the network input layer.
augimdsTrain = augmentedImageDatastore([224 224 3],imdsTrain);
augimdsValidation = augmentedImageDatastore([224 224 3],imdsValidation);

% Set training option
opts = trainingOptions("adam",...
    "MiniBatchSize",64,...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.001,...
    "MaxEpochs",10,...
    "Shuffle","every-epoch",...
    "ValidationFrequency",10,...
    "Plots","training-progress",...
    "ValidationData",augimdsValidation);

% clean up helper variable
clear tempLayers;

% Connect layer branches
lgraph = connectLayers(lgraph,"avg_pool","fc");

% Display layer
lgraph.Layers;
% figure
% plot(lgraph)
% title("DAG network")

% Train network (RUN ONLY ONCE)
% [net, traininfo] = trainNetwork(augimdsTrain,lgraph,opts);

% Save trained result (RUN ONLY ONCE)
% resnet50_result = net;
% save resnet50_result

% Load trained model
load resnet50_result
net = resnet50_result;

% Load test data
testData = imageDatastore("C:\Users\Billy\Desktop\Face_Recognition_DL\face_dataset_Test","IncludeSubfolders",true,"LabelSource","foldernames");

% Classify test data
[YPred, scores] = classify(net,testData);

% Confusion matrix
plotconfusion(testData.Labels,YPred)

