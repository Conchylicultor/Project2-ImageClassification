function [ classVoteTr classVoteTe ] = trainModelSVM( Tr, Te, labels, degreePoly )
%trainModelSVM Test with Svm classifier
    fprintf('Training SVM (binary)...\n');
    assert(length(labels) == 2, 'Error, trying to apply binary to multiclass');
    
    % Train Data
    X = Tr.normX;
    y = Tr.y;
   
    % Train an SVM Classifier using the radial basis kernel.
    % Auto find a scale value for the kernel function
    % Standardize the predictors
   
    % Linear
    %SVMModel = fitcsvm(X,y);

    % Polynomial
    SVMModel = fitcsvm(X,y,'KernelFunction','polynomial','PolynomialOrder', degreePoly);
   
    % Predict the data
    [classVoteTr, score] = predict(SVMModel, Tr.normX);
    [classVoteTe, score] = predict(SVMModel, Te.normX);
end

