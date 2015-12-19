clc;

load('recodingGlobalTe.mat');

allK = 2:10;
datasetSize = 100 - 100./allK;

x = datasetSize*60;
y = mean(globalEvaluationTe);
e = std(globalEvaluationTe);

errorbar(x,y,e);
title('Learning curve');
ylabel('BER');
xlabel('Number of training sample');
%curtick = get(gca, 'XTick');
%set(gca, 'XTickLabel', cellstr(num2str(curtick(:))));