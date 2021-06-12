clc; clear all; close all;

% Create layer graph
lgraph = layerGraph();

% Adding layer branch
tempLayers = [
    imageInputLayer([224 224 3],"Name","imageinput")
    convolution2dLayer([3 3],64,"Name","conv1","Stride",[2 2])
    reluLayer("Name","relu_conv1")
    maxPooling2dLayer([3 3],"Name","pool1","Stride",[2 2])
    convolution2dLayer([1 1],16,"Name","fire2-squeeze1x1")
    reluLayer("Name","fire2-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","fire2-expand1x1")
    reluLayer("Name","fire2-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","fire2-expand3x3","Padding",[1 1 1 1])
    reluLayer("Name","fire2-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire2-concat")
    convolution2dLayer([1 1],16,"Name","fire3-squeeze1x1")
    reluLayer("Name","fire3-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","fire3-expand1x1")
    reluLayer("Name","fire3-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","fire3-expand3x3","Padding",[1 1 1 1])
    reluLayer("Name","fire3-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire3-concat")
    maxPooling2dLayer([3 3],"Name","pool3","Padding",[0 1 0 1],"Stride",[2 2])
    convolution2dLayer([1 1],32,"Name","fire4-squeeze1x1")
    reluLayer("Name","fire4-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","fire4-expand1x1")
    reluLayer("Name","fire4-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","fire4-expand3x3","Padding",[1 1 1 1])
    reluLayer("Name","fire4-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire4-concat")
    convolution2dLayer([1 1],32,"Name","fire5-squeeze1x1")
    reluLayer("Name","fire5-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","fire5-expand3x3","Padding",[1 1 1 1])
    reluLayer("Name","fire5-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","fire5-expand1x1")
    reluLayer("Name","fire5-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire5-concat")
    maxPooling2dLayer([3 3],"Name","pool5","Padding",[0 1 0 1],"Stride",[2 2])
    convolution2dLayer([1 1],48,"Name","fire6-squeeze1x1")
    reluLayer("Name","fire6-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],192,"Name","fire6-expand3x3","Padding",[1 1 1 1])
    reluLayer("Name","fire6-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","fire6-expand1x1")
    reluLayer("Name","fire6-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire6-concat")
    convolution2dLayer([1 1],48,"Name","fire7-squeeze1x1")
    reluLayer("Name","fire7-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","fire7-expand1x1")
    reluLayer("Name","fire7-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],192,"Name","fire7-expand3x3","Padding",[1 1 1 1])
    reluLayer("Name","fire7-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire7-concat")
    convolution2dLayer([1 1],64,"Name","fire8-squeeze1x1")
    reluLayer("Name","fire8-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","fire8-expand3x3","Padding",[1 1 1 1])
    reluLayer("Name","fire8-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","fire8-expand1x1")
    reluLayer("Name","fire8-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire8-concat")
    convolution2dLayer([1 1],64,"Name","fire9-squeeze1x1")
    reluLayer("Name","fire9-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","fire9-expand1x1")
    reluLayer("Name","fire9-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","fire9-expand3x3","Padding",[1 1 1 1])
    reluLayer("Name","fire9-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire9-concat")
    dropoutLayer(0.5,"Name","drop9")
    convolution2dLayer([1 1],1000,"Name","conv10")
    reluLayer("Name","relu_conv10")
    globalAveragePooling2dLayer("Name","pool10")
    fullyConnectedLayer(6,"Name","fc")
    softmaxLayer("Name","prob")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

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

% Connect layer branches
lgraph = connectLayers(lgraph,"fire2-relu_squeeze1x1","fire2-expand1x1");
lgraph = connectLayers(lgraph,"fire2-relu_squeeze1x1","fire2-expand3x3");
lgraph = connectLayers(lgraph,"fire2-relu_expand3x3","fire2-concat/in2");
lgraph = connectLayers(lgraph,"fire2-relu_expand1x1","fire2-concat/in1");
lgraph = connectLayers(lgraph,"fire3-relu_squeeze1x1","fire3-expand1x1");
lgraph = connectLayers(lgraph,"fire3-relu_squeeze1x1","fire3-expand3x3");
lgraph = connectLayers(lgraph,"fire3-relu_expand3x3","fire3-concat/in2");
lgraph = connectLayers(lgraph,"fire3-relu_expand1x1","fire3-concat/in1");
lgraph = connectLayers(lgraph,"fire4-relu_squeeze1x1","fire4-expand1x1");
lgraph = connectLayers(lgraph,"fire4-relu_squeeze1x1","fire4-expand3x3");
lgraph = connectLayers(lgraph,"fire4-relu_expand3x3","fire4-concat/in2");
lgraph = connectLayers(lgraph,"fire4-relu_expand1x1","fire4-concat/in1");
lgraph = connectLayers(lgraph,"fire5-relu_squeeze1x1","fire5-expand3x3");
lgraph = connectLayers(lgraph,"fire5-relu_squeeze1x1","fire5-expand1x1");
lgraph = connectLayers(lgraph,"fire5-relu_expand3x3","fire5-concat/in2");
lgraph = connectLayers(lgraph,"fire5-relu_expand1x1","fire5-concat/in1");
lgraph = connectLayers(lgraph,"fire6-relu_squeeze1x1","fire6-expand3x3");
lgraph = connectLayers(lgraph,"fire6-relu_squeeze1x1","fire6-expand1x1");
lgraph = connectLayers(lgraph,"fire6-relu_expand3x3","fire6-concat/in2");
lgraph = connectLayers(lgraph,"fire6-relu_expand1x1","fire6-concat/in1");
lgraph = connectLayers(lgraph,"fire7-relu_squeeze1x1","fire7-expand1x1");
lgraph = connectLayers(lgraph,"fire7-relu_squeeze1x1","fire7-expand3x3");
lgraph = connectLayers(lgraph,"fire7-relu_expand3x3","fire7-concat/in2");
lgraph = connectLayers(lgraph,"fire7-relu_expand1x1","fire7-concat/in1");
lgraph = connectLayers(lgraph,"fire8-relu_squeeze1x1","fire8-expand3x3");
lgraph = connectLayers(lgraph,"fire8-relu_squeeze1x1","fire8-expand1x1");
lgraph = connectLayers(lgraph,"fire8-relu_expand1x1","fire8-concat/in1");
lgraph = connectLayers(lgraph,"fire8-relu_expand3x3","fire8-concat/in2");
lgraph = connectLayers(lgraph,"fire9-relu_squeeze1x1","fire9-expand1x1");
lgraph = connectLayers(lgraph,"fire9-relu_squeeze1x1","fire9-expand3x3");
lgraph = connectLayers(lgraph,"fire9-relu_expand1x1","fire9-concat/in1");
lgraph = connectLayers(lgraph,"fire9-relu_expand3x3","fire9-concat/in2");

% Display layer
lgraph.Layers;
% figure
% plot(lgraph)
% title("DAG network")

% Train network (RUN ONLY ONCE)
%[net, traininfo] = trainNetwork(augimdsTrain,lgraph,opts);

% Save trained result (RUN ONLY ONCE)
% mymodel_result = net;
% save mymodel_result

% Load trained model
load mymodel_result
net = mymodel_result;

% Load test data
testData = imageDatastore("C:\Users\Billy\Desktop\Face_Recognition_DL\face_dataset_Test","IncludeSubfolders",true,"LabelSource","foldernames");

% Classify test data
[YPred, scores] = classify(net,testData);

% Confusion matrix
plotconfusion(testData.Labels,YPred)



