clc;
figure;
hold on;

load('recodingGlobalTe.mat');
load('recodingGlobalTr.mat');

% allK = 2:10;
% datasetSize = 100 - 100./allK;
% x = datasetSize*60;

x = 0.02:0.005:0.065;

%y = mean(globalEvaluationTe);
%e = std(globalEvaluationTe

y = mean(g1te);
e = std(g1te);

errorbar(x,y,e);

%y = mean(globalEvaluationTr);
%e = std(globalEvaluationTr);
y = mean(g1tr);
e = std(g1tr);
errorbar(x,y,e);

title('Learning curve');
ylabel('BER');
xlabel('L2 penality');

legend('Testing','Training');
%curtick = get(gca, 'XTick');
%set(gca, 'XTickLabel', cellstr(num2str(curtick(:))));


