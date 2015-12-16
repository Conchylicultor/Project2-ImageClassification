function [ X_TrainAfter, X_TestAfter ] = dataTransform(X_Train, X_Test)
%dataTransform Test some basic maths transformation

    % Initialisation
    X_TrainAfter = [];
    X_TestAfter = [];

    for i=1:length(X_Train(1,:)) % For each collumn
        % We add the collumn and eventually apply a transformation
        X_TrainAfter = [X_TrainAfter basicMath(X_Train(:,i))];
        X_TestAfter = [X_TestAfter basicMath(X_Test(:,i))];
    end

end

function [Y] = basicMath(X)
    Y = X;
    %sqrt(abs(X));
    %Y = X.*X; % Much better
    %Y = X.*X.*X; % Much much better
    %Y = X.*X.*X.*X;
    
    %Y = exp(X); % Worst
    %Y = 1./X; % If X = 0 ???
    
%     if(abs(X) == X) % Only positive values
%         %Y = sqrt(X); % If X < 0 ???
%         %Y = log(X); % If X < 0 ???
%     else
%         Y = X;
%     end
end
