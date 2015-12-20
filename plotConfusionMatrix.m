function [] = plotConfusionMatrix( classVote, classReal )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

N = length(classVote);
targets = zeros(2,N);
outputs = zeros(2,N);

for k = 1:N
    targets(classReal(k)+1,k) = 1;
    outputs(classVote(k)+1,k) = 1;
end


plotconfusion(targets,outputs);

end

