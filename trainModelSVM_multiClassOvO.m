function [ classVoteTr classVoteTe ] = trainModelSVM_multiClassOvO( Tr, Te, labels )
%trainModelSVM Test with Svm classifier
    fprintf('Training SVM (multi class), OvsO...\n');
    assert(length(labels) > 2, 'Error, trying to apply multiclass to binary');
    
    nbClassifier = length(labels);
    nbClassifier = (nbClassifier*(nbClassifier-1))/2; % Six classifier
    
    fprintf('Training %d classifiers\n', nbClassifier);
    
    % Contain the predictions
    ScoresTr = zeros(length(Tr.normX(:,1)), length(labels));
    ScoresTe = zeros(length(Te.normX(:,1)), length(labels));
    
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
            SVMModel = fitcsvm(Xtrain,double(ytrain),'ClassNames',[0 1]);
            
            % Testing
            fprintf('Prediction class %d vs %d:\n', labels(i), labels(j));
            
            [class,~] = predict(SVMModel, Tr.normX);
            ScoresTr(class == 1, i) = ScoresTr(class == 1, i) + 1; % Winner is i
            ScoresTr(class == 0, j) = ScoresTr(class == 0, j) + 1; % Otherwise winner is j
            
            [class,~] = predict(SVMModel, Te.normX);
            ScoresTe(class == 1, i) = ScoresTe(class == 1, i) + 1; % Winner is i
            ScoresTe(class == 0, j) = ScoresTe(class == 0, j) + 1; % Otherwise winner is j
            
            % Save the svm ?
            %save([], );
        end
    end
    
    % Choose the best class
    [~,classVoteTe] = max(ScoresTe,[],2);
    [~,classVoteTr] = max(ScoresTr,[],2);
    
    %Scores(1:20,:)
    %[Te.y(1:20,:) classVote(1:20,:)]
        
end

