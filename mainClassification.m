%% Initialize variables

clearvars;
close all;
clc;

% Load features and labels of training data
fprintf('Loading data...\n');
load train/train.mat;

addpath(genpath('DeepLearnToolbox/'));
addpath(genpath('PiotrToolbox/'));

% Do we put the seed before or after the random sorting ?
% rng(8339);  % fix seed, this NN may be very sensitive to initialization

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

%% Feature transformations

% Apply PCA

% TODO: ALSO TRANSFORM TEST DATA !!!!!

% Data transformations on the features HoG and CNN directly
%train.X_cnn = dataTransform(train.X_cnn);
%train.X_hog = dataTransform(train.X_hog);

% size(train.X_cnn)
% pcaToSave = train.X_cnn;
% save('afterPca', 'pcaToSave');

%% Divide our dataset bewteen training/testing set AND select features

fprintf('Splitting into train/test..\n');

Tr = [];
Te = [];

% Randomly permute the data
idx = randperm(size(train.X_cnn,1));
train.X_cnn = train.X_cnn(idx, :);
train.X_hog = train.X_hog(idx, :);
train.y     = train.y    (idx);


% Choose our features
%train.features = [train.X_hog]; % Only HoG
train.features = [train.X_cnn]; % Only CNN (Seems to work better)
%train.features = [train.X_cnn train.X_hog]; % CNN + Hog

% Normalisation done inside the cross-validation (warning: we need to normalize HoG and CNN independently)

clear train.X_cnn;
clear train.X_hog; % Free some memory (not needed anymore)


%% Binary or multiclass classification

taskBinary=false;
if taskBinary == true
    disp('------ Binary mode ------');
    %nbLabel=2; % Only two class
    labels = [0 1];
    
    train.y(train.y ~= 4) = 1;
    train.y(train.y == 4) = 0; % -1 ?? Or 0, Or 2 ??
else
    disp('------ Multi-class mode ------');
    %nbLabel=4; % Four differents class
    labels = [1,2,3,4];
end

%% Train our model (evaluated with cross-validation)

% Select the k-fold


globalEvaluationTe = []; % Store our results
globalEvaluationTr = []; % Store our results

for param = 2:10 % Choose the param for which we are doing the cross validation !!!!!!
    fprintf('Test param %f:\n', param);
    
    % Here we test the number of training sample (k_fold)
    k_fold = param;
    ind = crossvalind('Kfold', size(train.X_cnn,1), k_fold);
    
    
    currentEvaluationTe = [];
    currentEvaluationTr = [];
    for k=1:k_fold
        fprintf('Kfold %d/%d:\n', k, k_fold);
        % Select training/testing

        % Generate train and test data : Dividing in two groups train and test set
        kPermIdx = (ind~=k);

        % Get the train set
        Tr.X = train.features(kPermIdx,:); % Data (features)
        Tr.y = train.y(kPermIdx,:); % Labels

        % Get the test set
        Te.X  = train.features(~kPermIdx,:);
        Te.y  = train.y(~kPermIdx,:);
        Te.idxs = idx(~kPermIdx);


        % Normalize data !!!

        %[Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std
        %Te.normX = normalize(Te.X, mu, sigma);  % normalize test data
        %Te.normX = Te.X / 64; % Simple normalization NON !!!
        Tr.normX = Tr.X;
        Te.normX = Te.X;

        % Form the tX
        Tr.normX = [ones(length(Tr.y), 1) Tr.normX];
        Te.normX = [ones(length(Te.y), 1) Te.normX];

        % Train and test our model
        %Te.predictions = trainModelNN(Tr, Te, labels);
        %Te.predictions = trainModelSVM(Tr, Te, labels);
        [Tr.predictions, Te.predictions] = trainModelSVM_multiClassOvO(Tr, Te, labels);
        %Te.predictions = trainModelSVM_multiClassOvA(Tr, Te, labels);
        %Te.predictions = trainModelIRLS(Tr, Te, labels);

        % Get and plot the errors
        fprintf('\n');
        
        [berErr, MatrixError] = computeBER(Tr.predictions , Tr.y, labels); % BER Error
        predErr = sum( Tr.predictions ~= Tr.y ) / length(Tr.y); % Overall error
        currentEvaluationTr = [currentEvaluationTr ; berErr];
        fprintf('Training error: %.2f%%, %.2f%% (Ber)\n', predErr * 100, berErr);
        
        predErr = sum( Te.predictions ~= Te.y ) / length(Te.y); % Overall error
        [berErr, MatrixError] = computeBER(Te.predictions , Te.y, labels); % BER Error
        currentEvaluationTe = [currentEvaluationTe ; berErr];

        disp(MatrixError);

        fprintf('Testing error: %.2f%%, %.2f%% (Ber)\n', predErr * 100, berErr);
        disp (['Nb of error: ', num2str(sum(Te.predictions ~= Te.y)), '/', num2str(length(Te.predictions))]);

        % Plot the errors images    
        %visualizeResults(Te);

        % Only one (TODO: remove in the final version)
        %return;
        
        save('recodingCurrentTr', 'currentEvaluationTr');
        save('recodingCurrentTe', 'currentEvaluationTe');
    end
    disp(currentEvaluationTe);
    disp(['Current eval: ', num2str(mean(currentEvaluationTe)) , ' +/- ', num2str(std(currentEvaluationTe))]) % Disp current evaluation

    globalEvaluationTe = [globalEvaluationTe currentEvaluationTe]; % Add the current evaluation
    save('recodingGlobalTe', 'globalEvaluationTe');
    
    globalEvaluationTr = [globalEvaluationTr currentEvaluationTr]; % Add the current evaluation
    save('recodingGlobalTr', 'globalEvaluationTr');

end

boxplot(globalEvaluationTe); % Plot all our evaluations
%errorbar(x,y,e);

%% End
return;
