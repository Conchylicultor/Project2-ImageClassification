function [ classVote ] = trainModelNN( Tr, Te, labels )
%trainModelNN Simple neural network with one hidden layers
    fprintf('Training simple neural network..\n');

    % setup NN. The first layer needs to have number of features neurons,
    %  and the last layer the number of classes (here four).
    nn = nnsetup([size(Tr.X,2) 15 length(labels)]);
    opts.numepochs =  15;   %  Number of full sweeps through data
    opts.batchsize = 100;  %  Take a mean gradient step over this many samples

    % if == 1 => plots trainin error as the NN is trained
    opts.plot               = 1;

    nn.learningRate = 2;

    % this neural network implementation requires number of samples to be a
    % multiple of batchsize, so we remove some for this to be true.
    numSampToUse = opts.batchsize * floor( size(Tr.X) / opts.batchsize);
    Tr.X = Tr.X(1:numSampToUse,:);
    Tr.y = Tr.y(1:numSampToUse);

    % prepare labels for NN
    LL = [];
    for i=1:length(labels)
        LL = [LL, 1*(Tr.y == labels(i))]; % first column, p(y=1)
                                          % second column, p(y=2), etc
    end

    [nn, L] = nntrain(nn, Tr.normX, LL, opts);

    % to get the scores we need to do nnff (feed-forward)
    %  see for example nnpredict().
    % (This is a weird thing of this toolbox)
    nn.testing = 1;
    nn = nnff(nn, Te.normX, zeros(size(Te.normX,1), nn.size(end)));
    nn.testing = 0;


    % predict on the test set
    nnPred = nn.a{end};

    % get the most likely class
    [~,classVotePred] = max(nnPred,[],2);

    classVote = zeros(size(classVotePred));
    for i=1:length(labels)
        classVote(classVotePred == i) = labels(i);
    end
end

