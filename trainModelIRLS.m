function [ predictionClass ] = trainModelIRLS( Tr, Te, labels )
%trainModelIRLS use the Iterative Recursive Least-Squares algorithm
    fprintf('Training using IRLS..\n');

    % For now, works only for binary classification
    
    betaIrls = computeIRLS(Tr.y, Tr.normX, labels);
    
    predictionClass = predictClass(Te.normX, betaIrls)
end

function [ beta ] = computeIRLS( y, tX , labels)
%computeIRLS use the Iterative Recursive Least-Squares algorithm

    % parametres
    maxIters = 10;
    beta = sparse(zeros(length(tX(1,:)),1)); 

    % Conversions to sparse
    y = sparse(double(y));
    tX = sparse(double(tX));
    
    for k = 1:maxIters     
    disp('w');
        sig = sparse(sigmoid(tX*beta));
    disp('x');
        s = sparse(sig.*(1-sig));
    disp('c');
        z = sparse(tX*beta + (y-sig)./s);
    disp('v');
        beta = weightedLeastSquares(z,tX,s);
    disp('b');

        [berError, ~] = computeBER(predictClass(tX,beta), Tr.y, labels);
        disp(['Iter:', num2str(berError)]);
    end

end

function [ beta ] = weightedLeastSquares( z, tX, s )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    %Xt X z
    disp('a');
    S = sparse(diag(s));
    disp('b');
    M1 = tX'*S*z; 
    disp('c');
    % XtsX
    M2 = tX'*S*tX;
    disp('e');
    
    %M2^-1 M1
    beta = M2 \ M1;
    
end

function [y] = sigmoid(x)
%sigmoid Compute sigma(x)
    y = 1 ./ (1+exp(-x));
end

function [predictions] = predictClassBlured(tX, beta)
%predictClassificationModel predict the class from tX and beta
    predictions = sigmoid(tX*beta)
end

function [predictions] = predictClass(tX, beta)
%predictClassificationModel predict the class from tX and beta
    predictions = +(predictClassBlured(tX, beta) > 0.5)
end
