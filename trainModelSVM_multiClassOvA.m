function [ classVoteTr classVoteTe ] = trainModelSVM_multiClassOvA( Tr, Te, labels )
%trainModelSVM Test with Svm classifier
    fprintf('Training SVM (multi class), OvA...\n');
    assert(length(labels) > 2, 'Error, trying to apply multiclass to binary');
    
    nbClassifier = length(labels);
    
    fprintf('Training %d classifiers\n', nbClassifier);

    ScoresTr = zeros(length(Tr.normX(:,1)), nbClassifier);
    ScoresTe = zeros(length(Te.normX(:,1)), nbClassifier);
    
    % Train 2by2 SVM Classifier
    for j = 1:nbClassifier;
        % Train 2by2 SVM Classifier
        fprintf('Start training class %d\n', labels(j));
        indx = 1*(Tr.y == labels(j)); % Create binary classes for each classifier
        SVMModel = fitcsvm(Tr.normX,indx,'ClassNames',[0 1]);
        
        % Predict the data
        fprintf('Prediction class %d:\n', labels(j));
        
        [class,score] = predict(SVMModel, Tr.normX);
        ScoresTr(:,j) = score(:,2); % Second column contains positive-class scores
        
        [class,score] = predict(SVMModel, Te.normX);
        ScoresTe(:,j) = score(:,2); % Second column contains positive-class scores
    end
   
    %Te.y(1:20,:)
    
    % Choose the best class
    [maxValue,classVoteTr] = max(ScoresTr,[],2);
    [maxValue,classVoteTe] = max(ScoresTe,[],2);
    
    %ScoresTe(1:20,:)
    %maxValue(1:20,:)
        
end

