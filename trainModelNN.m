function [ classVoteTr, classVoteTe ] = trainModelNN( Tr, Te, labels, param )
%trainModelNN Simple neural network with one hidden layers
    fprintf('Training simple neural network..\n');

    % setup NN. The first layer needs to have number of features neurons,
    %  and the last layer the number of classes (here four).
    nn = nnsetup([size(Tr.normX,2) 100 length(labels)]);
    opts.numepochs =  20;   %  Number of full sweeps through data
    opts.batchsize = 100;  %  Take a mean gradient step over this many samples

    % if == 1 => plots trainin error as the NN is trained
    opts.plot = 1;
    
    nn.activation_function              = 'sigm';   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
    nn.learningRate                     = 2;            %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
    nn.momentum                         = param;          %  Momentum
    nn.scaling_learningRate             = 0.9;            %  Scaling factor for the learning rate (each epoch)
    nn.weightPenaltyL2                  = 0.02;            %  L2 regularization
    nn.nonSparsityPenalty               = 0.0;            %  Non sparsity penalty
    nn.sparsityTarget                   = 2.00;         %  Sparsity target
    nn.inputZeroMaskedFraction          = 0;            %  Used for Denoising AutoEncoders
    nn.dropoutFraction                  = 0;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
    nn.testing                          = 0;            %  Internal variable. nntest sets this to one.
    nn.output                           = 'sigm';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'


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
    nn = nnff(nn, Tr.normX, zeros(size(Tr.normX,1), nn.size(end)));
    nn.testing = 0;
    
    % predict on the test set
    nnPredTr = nn.a{end};
    
    nn.testing = 1;
    nn = nnff(nn, Te.normX, zeros(size(Te.normX,1), nn.size(end)));
    nn.testing = 0;
    
    nnPredTe = nn.a{end};

    % get the most likely class
    [~,classVotePredTr] = max(nnPredTr,[],2);
    [~,classVotePredTe] = max(nnPredTe,[],2);

    classVoteTr = zeros(size(classVotePredTr));
    classVoteTe = zeros(size(classVotePredTe));
    for i=1:length(labels)
        classVoteTr(classVotePredTr == i) = labels(i);
        classVoteTe(classVotePredTe == i) = labels(i);
    end
end

