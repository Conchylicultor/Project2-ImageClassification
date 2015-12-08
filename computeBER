function [ errorBER, tableError ] = computeBER( classVote, classReal, numberClass)
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


    Class = 1:numberClass;
    Error = zeros(1,numberClass);
    
    for c = Class
        Error(c) = sum((classVote == c).*(classVote ~= classReal))*100 / sum((classVote == c));
    end
    Class = Class';
    Error = Error';
    tableError = table(Class, Error);
    errorBER = sum(tableError.Error) / numberClass;
end
