
Ytest = Te.predictions;

tblTr = tabulate(Tr.y);
tblTe = tabulate(Ytest);

length(Tr.predictions)
length(Te.predictions)

size(Ytest)

bar([tblTr(:,3) tblTe(:,3)]);
title('Prediction vs Training sample');
xlabel('Class');
ylabel('Proportion');
h = legend('Training', 'Predicted');
h.Location = 'northwest';

colormap('summer');
%save('pred_binary', 'Ytest');
%save('pred_multiclass', 'Ytest');

