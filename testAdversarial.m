function Q = testAdversarial()

addpath('minConf/minFunc');
addpath('minConf/minConf');
addpath('minConf');
load data


[nc, nn, nf] = size(x) % ??? do we need to reshape?

Q_init = zeros(nn+1, nc); % add alpha to q
%init q
for i = 1:nc
    Q_init(:,i) = rand(1,nn+1);
end

%objective
funObj = @(q)adversarialRankingQObj(reshape(q,[nn+1, nc]), x, u, lamda, mu); %reshape q??

%projection
funProj = @(q)projectQBisthochastic(reshape(q,[nn+1, nc]));   %reshape q??

%run, find q

q_init = Q_init(:);
%save('data1.mat','q_init','-append');

options = []
options.maxIter = 500;
options.optTol = 1e-10;
options.projTol = 1e-10;
options.progTol = 1e-15;



q_opt = minConf_SPG(funObj, q_init, funProj, options);
Q_opt = reshape(q_opt, [nn+1, nc]); 
Q = Q_opt

% tetha = minTheta(..) we can find it in python code!


end

function [f, g] = adversarialRankingQObj(q, x, u, lamda, mu)


q = transpose(q)
save('data.mat','q','-append')


temp = py.adv_test.adversarialRankingTestObj() %temp = py.adv_train2.adversarialRankingObj()  % pass data using .mat file

load data1
load data2
g = transpose(g)
f = f

end

function [vz] = projectQBisthochastic(Q)
%Q: 25*500

[nn, nc] = size(Q);
Z = zeros(nn, nc);
for i = 1:nc
    Qi = Q(:,i);
    Zi = projectSthochasticADMM(Qi); %Zi = projectBisthochasticADMM(Qi);
    Z(:,i) = Zi;
end

%vz = Z(:);
vz = Q(:);

end

function Z  = projectADMMpython() % X: 25*25

load data5
X = X_data;
Z_init = Z_data;
Z = projectBisthochasticADMM( X, Z_init);

end

function [ Z ] = projectSthochasticADMM( X, Z_init ) % X: 26*1 temporary function % implement this for q and alpha
nn = size(X);
Z = X;
for i = 1:(nn-1) % check if  0<q<1
    if Z(i) < 0
        Z(i) = 0;
    else if Z(i) > 1
        Z(i) = 1;
    end
end

if Z(nn) < 0   % check if alpha > 0
    Z(nn) = 0;
end
end %too many end!!
end



