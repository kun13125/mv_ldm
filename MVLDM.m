function [Predict_Y1,Predict_Y2,Predict_Y] = MVLDM(Test,DataTrain,FunPara)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initailization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Xa = DataTrain.Xa;
    Xb = DataTrain.Xb;
    y = DataTrain.Y;
    Xa1 = Xa(y>0,:);     
    Xa2 = Xa(y<0,:);       
    Xb1 = Xb(y>0,:);
    Xb2 = Xb(y<0,:);
    c1 = FunPara.c1;
    c3 = FunPara.c3;
    a1 = FunPara.a1;
    a2 = FunPara.a2;
    a3 = FunPara.a3;
    kerfPara = FunPara.kerfPara;
    % 
    na = size(Xa,1);
    na1 = size(Xa1,1);
    na2 = size(Xa2,1);
    nb = size(Xb,1);
    nb1 = size(Xb1,1);
    nb2 = size(Xb2,1);
    ma = size(Xa,2);   
    mb = size(Xb,2);
    %
    y = [ones(na1,1);-1*ones(na2,1)];
    O1 = zeros(na,mb);                                       
    O2 = zeros(nb,ma); 
    Xa = [Xa1;Xa2];
    Xb = [Xb1;Xb2];
    X = [Xa,O1;O2,Xb];
    X1 = [Xa,Xb];
    Y = diag([ones(na1,1);-1*ones(na2,1);ones(nb1,1);-1*ones(nb2,1)]); 
    Y2 = [eye(na),-1*eye(na)];
    L = diag([ones(ma,1);a3*ones(mb,1)]);           
    OO3 = zeros(na,na);
    B = -(a1+a2)*y*y'/(2*na*na);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Kernel 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if strcmp(kerfPara.type,'lin')
        Q = L^-1+a1*X'*X/na+X1'*B*X1;
        HHH = Q\X';
        HH = X*HHH;
    else
        Ga = kernelfun(Xa,kerfPara,Xa);                 
        Gb = kernelfun(Xb,kerfPara,Xb);
        XX = [Ga,OO3;OO3,Gb/a3];
        XX1 = [Ga;Gb/a3]; 
        X1X = [Ga,Gb/a3];
        X1X1 = Ga+Gb/a3;
        BB = na*eye(na+nb)/a1+XX;
        H1 = XX-XX*(BB\XX);
        H2 = XX1-XX*(BB\XX1);
        H3 = X1X1-X1X*(BB\XX1);
        H4 = X1X-X1X*(BB\XX);
        HHH = (B+B*H3*B+0.000001*eye(na))\B;       
        HH = H1-H2*B*HHH*H4;
    end
    YY = [Y,-1*Y2',Y2'];
    H = YY'*HH*YY;
    M = -[ones(na+nb,1);-1*c3*ones(na,1);-1*c3*ones(na,1)]';
    b = c1*ones(3*na,1);
    OO = 0*eye(na);
    E = [OO;OO;eye(na)];                              
    A = [eye(3*na),E];
%%%%    ADMM   %%%%%%%%
    p = 1;  
    T = [A;-eye(4*na)];
    Q = [c1*ones(3*na,1);0*ones(4*na,1)];
    iter = 50;
    pai = ADMM(H,M,p,T,Q,iter);   
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    Predict
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nT = size(Test.Xa,1);
    OO6 = zeros(nT,na);
    if strcmp(kerfPara.type,'lin')
        w = HHH*YY*pai;
        y1 = Test.Xa*w(1:ma);
        y2 = Test.Xb*w(ma+1:end);
        y3 = [Test.Xa,Test.Xb*a3]*w;
    else
        XTa = kernelfun(Test.Xa,kerfPara,Xa);
        XTb = kernelfun(Test.Xb,kerfPara,Xb);
        XTX = [XTa,OO6;OO6,XTb];   
        XTX1 = [XTa;XTb];
        HT1 = XTX-XTX*(BB\XX);
        HT2 = XTX1-XTX*(BB\XX1);
        y = (HT1-HT2*B*HHH*H4)*YY*pai;
        y1 = y(1:nT);
        y2 = y(nT+1:end);
        y3 = y(1:nT)/2+y(nT+1:end)/2;                 
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    Predict_Y1 = sign(y1);
    Predict_Y2 = sign(y2);
    Predict_Y = sign(y3);
end
