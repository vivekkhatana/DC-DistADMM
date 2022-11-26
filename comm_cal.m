% This code is written for simulations of gradient consensus (journal version) 

clear all

%% graph initialization
load('graphArray_100Nodes_100ConnectedDiGraph');
%load('graphArray_20Nodes_10000ConnectedDiGraph');
% load('graphArray_100Nodes_1000ConnectedDiGraph');
Num_graphs = size(arr,1);
numberNodes = sqrt(size(arr,2));

Diameter = 5;

% numGraphs = 100;
% numIniCond = 100;

%% Choose the objective function
index = 1; % index = 1 :Strongly Conv (Quadratic Prob), index = 2 :Conv(Least Squares), index = 3: Logistic Reg 
convergence_Tol = 0.01;

%%% Quadratic Optimization
Dimension = 10; % , numberNodes  

%%% Least Squares
NumofFeatures = 10; % , numberNodes
Num_Examples = 100;

%%% Logistic Regression
Training_Examples = 20;
Num_Features = 15;


vec_Dim = NumofFeatures;

%% Problem data initialization
[A, B, IniEstimate, C, Lh, Lfhat, mu, x_true] = Initialize_Test( numberNodes, vec_Dim, Num_Examples, Training_Examples, index );


%% CVX code to solve the problem centrally

[xstar, fstar] = Centralized_CVX(vec_Dim, A, B, C, mu, numberNodes, index, x_true);

Xstar = xstar.*ones(vec_Dim,numberNodes);

%% Distributed implementation
% stepsize = [0.5, 0.1, 0.05, 0.01];
k = 3;

gm = 1e1;

alpha = 0.005;
Max_Iter = 50*Diameter;

%% initialization 
COMM_EXTRA = []; COMM_DGD = []; COMM_DDistADMM = []; COMM_PushPull = []; COMM_WShiADMM = []; COMM_MultiAgentADMM = []; 


for i = 1:Num_graphs
    
    currentG = arr(i,:);   
    currentG = reshape(currentG,numberNodes,numberNodes)'+eye(numberNodes);
    Weight_Doub_Stoc = PDoubleStochastic(currentG,numberNodes);
    Weight_Row_Stoc = PRowStochastic(currentG,numberNodes); 

    %% Problem data initialization
    [A, B, IniEstimate, C, Lh, Lfhat, mu, x_true] = Initialize_Test( numberNodes, vec_Dim, Num_Examples, Training_Examples, index );


    %% CVX code to solve the problem centrally

    [xstar, fstar] = Centralized_CVX(vec_Dim, A, B, C, mu, numberNodes, index, x_true);

    Xstar = xstar.*ones(vec_Dim,numberNodes);

    
    [sol_res_DDistADMM, objval_err_DDistADMM, comm_DDistADMM, constraint_resk] = DDistADMM(vec_Dim, numberNodes,Weight_Doub_Stoc, A, B, ...
                                                                         IniEstimate, C, Diameter, ...
                                                                         gm, index, Xstar, fstar,Max_Iter, mu, Lh, k);
    COMM_DDistADMM = [COMM_DDistADMM comm_DDistADMM];
%     

%     [sol_res_DGD, objval_err_DGD, comm_DGD] = DGD(numberNodes,...
%                                                           Weight_Doub_Stoc, A, B, IniEstimate, C, ...
%                                                           alpha, index, Xstar, fstar, Max_Iter, mu);
%     COMM_DGD = [COMM_DGD comm_DGD];
    
    

%     [sol_res_EXTRA, objval_err_EXTRA, comm_EXTRA] = EXTRA(numberNodes,...
%                                                           Weight_Doub_Stoc, A, B, IniEstimate, C, ...
%                                                           alpha, index, Xstar, fstar,Max_Iter, mu);
%     COMM_EXTRA = [COMM_EXTRA comm_EXTRA];

    

%     [sol_res_PushPull, objval_err_PushPull, comm_PushPull] = PushPull(numberNodes,...
%                                                           Weight_Row_Stoc, A, B, IniEstimate, C, ...
%                                                           alpha, index, Xstar, fstar, Max_Iter, mu);
%     COMM_PushPull = [COMM_PushPull comm_PushPull];
    


%     [sol_res_WShiADMM, objval_err_WShiADMM, comm_WShiADMM] =  WShiADMM(numberNodes,...
%                                                           currentG, A, B, IniEstimate, C,...
%                                                           index, Xstar, fstar, Max_Iter, mu);                                                 
%     COMM_WShiADMM = [COMM_WShiADMM comm_WShiADMM];
    

%     [sol_res_MultiAgentADMM, objval_err_MultiAgentADMM, comm_MultiAgentADMM] = MultiAgentADMM(numberNodes,...
%                                                           currentG, A, B, IniEstimate, C,...
%                                                           index, Xstar, fstar,Max_Iter,mu);
%     COMM_MultiAgentADMM = [COMM_MultiAgentADMM comm_MultiAgentADMM];
    
end
    %% Plotting 
%     figure(1);
%     histogram(COMM_EXTRA,'Normalization','probability','BinMethod','scott','DisplayStyle','bar','EdgeAlpha',0.5) 
%     legend('EXTRA')
%     xlabel('Number of Communication Steps');
%     ylabel('Empirical Distribution');

%     figure(2);
%     histogram(COMM_DGD,'Normalization','probability','BinMethod','scott','DisplayStyle','bar','EdgeAlpha',0.5)
%     legend('DGD')
%     xlabel('Number of Communication Steps');
%     ylabel('Empirical Distribution');

%     figure(3);
%     histogram(COMM_PushPull,'Normalization','probability','BinMethod','scott','DisplayStyle','bar','EdgeAlpha',0.5)
%     legend('PushPull')
%     xlabel('Number of Communication Steps');
%     ylabel('Empirical Distribution');

    figure(4);
    histogram(5*COMM_DDistADMM,'Normalization','probability','BinMethod','scott','DisplayStyle','bar','EdgeAlpha',0.5)
    legend('DDistADMM')
    xlabel('Number of Communication Steps');
    ylabel('Empirical Distribution');

%     figure(5);
%     histogram(COMM_WShiADMM,'Normalization','probability','BinMethod','scott','DisplayStyle','bar','EdgeAlpha',0.5)
%     legend('DCOADMM')    
%     xlabel('Number of Communication Steps');
%     ylabel('Empirical Distribution');

%     figure(6);
%     histogram(COMM_MultiAgentADMM,'Normalization','probability','BinMethod','scott','DisplayStyle','bar','EdgeAlpha',0.5)
%     legend('MultiAgentADMM')
%     xlabel('Number of Communication Steps');
%     ylabel('Empirical Distribution');
