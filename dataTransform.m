function [ X_After ] = dataTransform(X_Before)
%dataTransform Test some basic maths transformation

    % PCA
    coeff = pca(X);

    % Initialisation
    X_After = [];

    for i=1:length(X_Before(1,:)) % For each collumn
        X_After = [X_After basicMath(X_Before(:,i))];
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
    
    %Y = sqrt(abs(X));
    %Y = log(abs(X));
end
