function [ classVote ] = trainModelSVM( Tr, Te, labels )
%trainModelSVM Test with Svm classifier
    fprintf('Training SVM (binary)...\n');
    
    % Train Data
    X = Tr.normX;
    y = Tr.y;
   
    % Train an SVM Classifier using the radial basis kernel.
    % Auto find a scale value for the kernel function
    % Standardize the predictors
    SVMModel = fitcsvm(X,y,'KernelFunction','RBF','KernelScale','auto', 'BoxConstraint', Inf);
    
    % Cross validate the SVM Classifier
    % default : 10-fold cross validation 
    %CVSVMModel = crossval(SVMModel);
    
    % Estimate the out-of-sample missclassification rate
    %classLoss = kfoldLoss(CVSVMModel)
   
    % Predict the data
    [classVote, score] = predict(SVMModel, Te.normX);
end

