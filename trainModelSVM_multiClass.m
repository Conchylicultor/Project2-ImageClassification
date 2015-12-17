function [ classVote ] = trainModelSVM_multiClass( Tr, Te, labels )
%trainModelSVM Test with Svm classifier
    fprintf('Training SVM (multi class)...\n');
    assert(length(labels) > 2, 'Error, trying to apply multiclass to binary');
    
    nbClassifier = length(labels);
    
    fprintf('Training %d classifiers\n', nbClassifier);
    
    SVMModels = cell(nbClassifier,1);
    
    % Train 2by2 SVM Classifier
    for j = 1:nbClassifier;
        fprintf('Start training class %d\n', labels(j));
        indx = 1*(Tr.y == labels(j)); % Create binary classes for each classifier
        SVMModels{j} = fitcsvm(Tr.normX,indx,'ClassNames',[false true]);
    end
   
    fprintf('Predict results:\n');
    Te.y(1:20,:)
    
    % Predict the data
    Scores = zeros(length(Te.normX(:,1)), nbClassifier);
    %classOther = zeros(length(Te.normX(:,1)), 1); % Not classified
    for j = 1:nbClassifier;
        fprintf('Prediction class %d:\n', labels(j));
        [class,score] = predict(SVMModels{j}, Te.normX);
        class(1:20,:)
        Scores(:,j) = score(:,2); % Second column contains positive-class scores
    end
    
    % Choose the best class
    [maxValue,classVote] = max(Scores,[],2);
    
    Scores(1:20,:)
    maxValue(1:20,:)
        
end

