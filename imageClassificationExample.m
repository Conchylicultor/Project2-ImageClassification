%% Initialize variables

clearvars;
close all;

% Load features and labels of training data
load train/train.mat;

addpath(genpath('DeepLearnToolbox/'));
addpath(genpath('PiotrToolbox/'));

%
% rng(8339);  % fix seed, this    NN may be very sensitive to initialization

%% Some visualization --browse through the images and look at labels

visualisationActive = false;
if visualisationActive
    for i=1:10
        clf();

        % load img
        img = imread( sprintf('train/imgs/train%05d.jpg', i) );

        subplot(121);
        % show img
        imshow(img);
        title(sprintf('Label %d', train.y(i)));

        subplot(122);
        feature = hog( single(img)/255, 17, 8);
        im( hogDraw(feature) ); colormap gray;
        axis off; colorbar off;

        pause;  % wait for key, 
    end
end

%% Divide our dataset bewteen training/testing set AND select features

fprintf('Splitting into train/test..\n');

Tr = [];
Te = [];

% Randomly permute the data
idx = randperm(size(train.X_cnn,1));
train.X_cnn = train.X_cnn(idx, :);
train.X_hog = train.X_hog(idx, :);
train.y     = train.y    (idx);

% Select the k-fold
k_fold = 5;
ind = crossvalind('Kfold', size(train.X_cnn,1), k_fold);

% Choose our features
%train.features = [train.X_hog]; % Only HoG
train.features = [train.X_cnn]; % Only CNN (Seems to work better)
%train.features = [train.X_cnn train.X_hog]; % CNN + Hog

% TODO: Datatransform ?

% Normalisation done inside the cross-validation

%% Train our model (evaluated with cross-validation)

for k=1:k_fold
    fprintf('Training simple neural network..\n');
    
    % Select training/testing
    
    % Generate train and test data : Dividing in two groups train and test set
    kPermIdx = (ind~=k);
    
    % Get the train set
    Tr.X = train.features(kPermIdx,:); % Data (features)
    Tr.y = train.y(kPermIdx,:); % Labels
    
    % Get the test set
    Te.X  = train.features(~kPermIdx,:);
    Te.y  = train.y(~kPermIdx,:);

    
    % Normalize data !!!
    [Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std
    Te.normX = normalize(Te.X, mu, sigma);  % normalize test data
    
    

    % setup NN. The first layer needs to have number of features neurons,
    %  and the last layer the number of classes (here four).
    nn = nnsetup([size(Tr.X,2) 10 4]);
    opts.numepochs =  17;   %  Number of full sweeps through data
    opts.batchsize = 100;  %  Take a mean gradient step over this many samples

    % if == 1 => plots trainin error as the NN is trained
    opts.plot               = 1;

    nn.learningRate = 2;

    % this neural network implementation requires number of samples to be a
    % multiple of batchsize, so we remove some for this to be true.
    numSampToUse = opts.batchsize * floor( size(Tr.X) / opts.batchsize);
    Tr.X = Tr.X(1:numSampToUse,:);
    Tr.y = Tr.y(1:numSampToUse);

    % prepare labels for NN
    LL = [1*(Tr.y == 1), ...
          1*(Tr.y == 2), ...
          1*(Tr.y == 3), ...
          1*(Tr.y == 4) ];  % first column, p(y=1)
                            % second column, p(y=2), etc

    [nn, L] = nntrain(nn, Tr.normX, LL, opts);

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
    predErr = sum( classVote ~= Te.y ) / length(Te.y);

    fprintf('\nTesting error: %.2f%%\n\n', predErr * 100 );
    disp (['Nb of error: ', num2str(sum(classVote ~= Te.y)), '/', num2str(length(classVote))]);

end

%% End
return;

%% visualize samples and their predictions (test set)

% Plot the errors instead of sucess.
figure;
PlotErrorContinue = false;
nbFalseDetection = 0;
i = 1;
while nbFalseDetection < 3 % Will crash if we have less than 

    if train.y(Te.idxs(i)) ~= classVote(i) % Plot if different
        clf();
        
        img = imread( sprintf('train/imgs/train%05d.jpg', Te.idxs(i)) );
        imshow(img);

        % show if it is classified as pos or neg, and true label
        title(sprintf('Label: %d, Pred: %d', train.y(Te.idxs(i)), classVote(i)));
        
        nbFalseDetection = nbFalseDetection + 1;
        pause;  % wait for keydo that then, 
    end
    
    i = i+1;
end
