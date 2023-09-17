function pai = ADMM(H,M,p,T,Q,iter)

pai = 0*ones(size(H,1),1); 
Theta = 0*ones(size(Q,1),1);  
h = Theta;
k = 1;
HH = H'+p*T'*T;
R = zeros(iter,1);
PR = R;
DR = R;
pT = p*T';
while k <=iter
    
    Theta_old = Theta;
    V = Theta + h - Q;
    VV = - M' -  pT*V;   
    pai = CGsolver(HH,VV);  
    Tpai = T*pai;

    % update ¦È
    Theta = pos(Q-Tpai-h);     

    % update h
    pr = Tpai + Theta - Q; 
    h = h + pr;


    dr = pT*(Theta- Theta_old);       

    PR(k)  = norm(pr); 
    DR(k)  = norm(dr); 


    if  PR(k) <= 0.002 && DR(k) <= 0.002;
         k=k+1;
         break
    end     
    k=k+1;
end
fprintf('----------------------------------------------------------------------------------------------%d------------\n',k-1);

end
%%
function [x, niters] = CGsolver(A,b)
% cgsolve : Solve Ax=b by conjugate gradients
%

n = length(b);

tol=1e-4;
maxiters=20;

normb = norm(b);
x = zeros(n,1);
r = b;
rtr = r'*r;
d = r;
niters = 0;
while sqrt(rtr)/normb > tol  &&  niters < maxiters
    niters = niters+1;
    Ad = A*d;
    alpha = rtr / (d'*Ad);
    x = x + alpha * d;
    r = r - alpha * Ad;
    rtrold = rtr;
    rtr = r'*r;
    beta = rtr / rtrold;
    d = r + beta * d;
end
end
%%
function A = pos(A)
A(A<0)=0;
end