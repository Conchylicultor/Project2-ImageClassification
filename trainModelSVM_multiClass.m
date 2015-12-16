function [ classVote ] = trainModelSVM_multiClass( Tr, Te, labels )
%trainModelSVM Test with Svm classifier
    fprintf('Training SVM (multi class)...\n');
    
    nbClassifier = length(labels);
    
    fprintf('Training %d classifiers\n', nbClassifier);
    
    SVMModels = cell(nbClassifier,1);
    
    % Train 2by2 SVM Classifier
    for j = 1:nbClassifier;
        fprintf('Start training class %d\n', labels(j));
        %indx = strcmp(y,labels(j)*ones(size(y))); % Create binary classes for each classifier
        indx = 1*(Tr.y == labels(j)); % Create binary classes for each classifier
        SVMModels{j} = fitcsvm(Tr.normX,indx,'ClassNames',[false true],...
                               'KernelFunction','rbf','BoxConstraint',Inf); % TODO: What is boxconstraint ?
    end
   
    fprintf('Predict results:\n');
    
    % Predict the data
    Scores = zeros(length(Te.normX(:,1)), nbClassifier);
    %classOther = zeros(length(Te.normX(:,1)), 1); % Not classified
    for j = 1:nbClassifier;
        fprintf('Prediction class %b:\n', labels(j));
        [class,score] = predict(SVMModels{j}, Te.normX);
        class(1:20,:)
        Scores(:,j) = score(:,2); % Second column contains positive-class scores
    end
    
    % Choose the best class
    [maxValue,classVote] = max(Scores,[],2);
    
    Scores(1:20,:)
    maxValue(1:20,:)
        
end

