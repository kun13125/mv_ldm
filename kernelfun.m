function omega = kernelfun(Xtrain,kerfPara,Xt)
%%
kernel_type = kerfPara.type;
kernel_pars = kerfPara.pars;
nb_data = size(Xtrain,1);


if strcmp(kernel_type,'rbf'),
    if nargin<3,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        omega = exp(-omega./(2*kernel_pars(1)));
    else
        omega = - 2*Xtrain*Xt';
        Xtrain = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        Xt = sum(Xt.^2,2)*ones(1,nb_data);
        omega = omega + Xtrain+Xt';
        omega = exp(-omega./(2*kernel_pars(1)));
    end
  
elseif strcmp(kernel_type,'lin')
    if nargin<3,
        omega = Xtrain*Xtrain';
    else
        omega = Xtrain*Xt';
    end
    
end