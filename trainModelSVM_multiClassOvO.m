function [ classVote ] = trainModelSVM_multiClassOvO( Tr, Te, labels )
%trainModelSVM Test with Svm classifier
    fprintf('Training SVM (multi class), OvsO...\n');
    assert(length(labels) > 2, 'Error, trying to apply multiclass to binary');
    
    nbClassifier = length(labels);
    nbClassifier = (nbClassifier*(nbClassifier-1))/2; % Six classifier
    
    fprintf('Training %d classifiers\n', nbClassifier);
    
    % Contain the predictions
    Scores = zeros(length(Te.normX(:,1)), length(labels));
    
    % Train One vs One SVM Classifier
    for i = 1:length(labels);
        for j = (i+1):length(labels);
            fprintf('Start training %d vs %d\n', labels(i), labels(j));
            
            % Select the class
            indxI = Tr.y == labels(i);
            indxJ = Tr.y == labels(j);
            Xtrain = Tr.normX(bitor(indxI, indxJ),:);
            ytrain = Tr.y(bitor(indxI, indxJ));
            ytrain(ytrain == labels(i)) = 1;
            ytrain(ytrain == labels(j)) = 0;
            
            % Plot some infos
            %indxI(1:20)
            %indxJ(1:20)
            %Tr.y(1:20)
            %ytrain(1:20)
            
            % Training
            SVMModel = fitcsvm(Xtrain,double(ytrain),'ClassNames',[false true]);
            
            % Testing
            fprintf('Prediction class %d vs %d:\n', labels(i), labels(j));
            
            [class,~] = predict(SVMModel, Te.normX);
            
            Scores(class == 1, i) = Scores(class == 1, i) + 1; % Winner is i
            Scores(class == 0, j) = Scores(class == 0, j) + 1; % Otherwise winner is j
            
            % Save the svm ?
            %save([], );
        end
    end
    
    % Choose the best class
    [~,classVote] = max(Scores,[],2);
    
    Scores(1:20,:)
    [Te.y(1:20,:) classVote(1:20,:)]
        
end

