function [ X_After ] = dataTransform(X_Before)
%dataTransform Test some basic maths transformation

    % PCA
    disp('Apply PCA');
    
    size(X_Before)
    coeff = pca2(X_Before'); % Call inside the deep toolbox
    size(coeff)
    X_After = X_Before * coeff; % maxDim = 5000
    %X_After = X_Before * coeff(1:(size(coeff,2)/2));

    % Initialisation
%     X_After = [];
% 
%     for i=1:length(X_Before(1,:)) % For each collumn
%         X_After = [X_After basicMath(X_Before(:,i))];
%     end

end

function [newX] = myPCA(X, dimAfter)
    disp('1');
    X = sparse(double(X));
    disp('2');
    C = cov(X'*X);
    disp('3');
    [V, D] = eigs(C);
    disp('4');
    [~, order] = sort(diag(D), 'descend');       %# sort cols high to low
    disp('5');
    V = V(:,order);
    disp('6');
    newX = X*V;
    disp('7');
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
