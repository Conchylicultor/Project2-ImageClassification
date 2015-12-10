function [ errorBER, tableError ] = computeBER( classVote, classReal, labels)
%BER Compute the Balanced Error Rate of the classVote 
% USAGE
%  [ BER, MatrixError ] = BER( classVote, classReal, numberClass)
%
% INPUTS
%  classVote   - vector of the prediction class
%  classReal   - vector of the real class
%  numberClass - 
%
% OUTPUTS
%  BER            - BER error of the classVote
%  MatrixError    - Matrix return the error of each class


    numberClass = length(labels);
    
    Class = 1:numberClass;
    Error = zeros(1,numberClass);
    
    for c = Class
        Error(c) = sum((classVote == labels(c)).*(classVote ~= classReal))*100 / sum((classVote == labels(c)));
    end
    Error = Error';
    tableError = table(labels', Error);
    errorBER = sum(tableError.Error) / numberClass;
end
