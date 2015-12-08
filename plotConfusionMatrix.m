function [] = plotConfusionMatrix( classVote, classReal )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

N = length(classVote);
targets = zeros(4,N);
outputs = zeros(4,N);

for k = 1:N
    targets(classReal(k),k) = 1;
    outputs(classVote(k),k) = 1;
end


plotconfusion(targets,outputs);

end

