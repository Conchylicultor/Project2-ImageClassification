clearvars;
clear all; close all; clc;

% Load features and labels of training data
load train/train.mat;


%% Randomly permute the data and k-fold
fprintf('Splitting into train/test..\n');

Tr = [];
Te = [];

% Randomly permute the data
idx = randperm(size(train.X_cnn,1));
train.X_cnn = train.X_cnn(idx, :);
train.y = train.y(idx);

% k-fold
k_fold = 5;
ind = crossvalind('Kfold', size(train.X_cnn,1), k_fold);

for k = 1:k_fold
    fprintf('Dividing in two groups train and test set k = %d\n',k);
    % Generate train and test data : Dividing in two groups train and test set
    kPermIdx = (ind~=k);
    
    % Get the train set
    Tr.Data = train.X_cnn(kPermIdx,:);
    Tr.labels = train.y(kPermIdx,:);
    
    % Get the test set
    Te.Data  = train.X_cnn(~kPermIdx,:);
    Te.labels  = train.y(~kPermIdx,:);
    
    %% Perform neural network
    fprintf('Training simple neural network..\n');
    
    rng(8339);  % fix seed, this    NN may be very sensitive to initialization
    
    % setup NN. The first layer needs to have number of features neurons,
    %  and the last layer the number of classes (here four).
    nn = nnsetup([size(Tr.Data,2) 10 4]);
    
    opts.numepochs =  20;   %  Number of full sweeps through data
    opts.batchsize = 100;  %  Take a mean gradient step over this many samples
    
    % if == 1 => plots trainin error as the NN is trained
    opts.plot               = 1;
    
    nn.learningRate = 2;
    % this neural network implementation requires number of samples to be a
    % multiple of batchsize, so we remove some for this to be true.
    numSampToUse = opts.batchsize * floor( size(Tr.Data) / opts.batchsize);
    Tr.Data = Tr.Data(1:numSampToUse,:);
    Tr.labels = Tr.labels(1:numSampToUse);
    
    % normalize data
    [Tr.normX, mu, sigma] = zscore(Tr.Data); % train, get mu and std
    
    % prepare labels for NN
    LL = [1*(Tr.labels == 1), ...
        1*(Tr.labels == 2), ...
        1*(Tr.labels == 3), ...
        1*(Tr.labels == 4) ];  % first column, p(y=1)
    % second column, p(y=2), etc
    
    [nn, L] = nntrain(nn, Tr.normX, LL, opts);
    
    
    Te.normX = normalize(Te.Data, mu, sigma);  % normalize test data
    
    % to get the scores we need to do nnff (feed-forward)
    %  see for example nnpredict().
    % (This is a weird thing of this toolbox)
    nn.testing = 1;
    nn = nnff(nn, Te.normX, zeros(size(Te.normX,1), nn.size(end)));
    nn.testing = 0;
    
    
    % predict on the test set
    nnPred = nn.a{end};
    
    % get the most likely class
    [~,classVote] = max(nnPred,[],2);
    
    % get overall error [NOTE!! this is not the BER, you have to write the code
    %                    to compute the BER!]
    predErr = sum( classVote ~= Te.labels ) / length(Te.labels);
    
    %% Performance Evaluation BER
    BER = 0;
    for c = 1:4
       BER = BER + sum((classVote == c).*(classVote ~= Te.labels)) / length((classVote == c)); 
    end
    BER = BER / 4;
    fprintf('\nTesting error: %.2f%%\n\n', predErr * 100 );
    fprintf('\nBER error: %.2f%%\n\n', BER * 100 );
end

