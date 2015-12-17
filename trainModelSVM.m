function [ classVote ] = trainModelSVM( Tr, Te, labels )
%trainModelSVM Test with Svm classifier
    fprintf('Training SVM (binary)...\n');
    
    % Train Data
    X = Tr.normX;
    y = Tr.y;
   
    % Train an SVM Classifier using the radial basis kernel.
    % Auto find a scale value for the kernel function
    % Standardize the predictors
   
    % Linear
    %SVMModel = fitcsvm(X,y);

    % Polynomial
    SVMModel = fitcsvm(X,y,'KernelFunction','polynomial','PolynomialOrder',2);
   
    % Predict the data
    [classVote, score] = predict(SVMModel, Te.normX);
end

