function G = mykernelfunction(U,V)
%%
% Sigmoid kernel function with slope gamma and intercept c
gamma = 0.5;
c = -0.2;
G = tanh(gamma*U*V' + c);

