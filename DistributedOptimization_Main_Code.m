% This code is written for simulations of gradient consensus (journal version) 

clear all;

%% graph initialization
% load('graphArray_20Nodes_10000ConnectedDiGraph');
load('graphArray_100Nodes_1000ConnectedDiGraph');
% load('graphArray_100Nodes_1_D14ConnectedDiGraph');
Num_graphs = size(arr,1);
numberNodes = sqrt(size(arr,2));
graphNo = 663;
currentG = arr(graphNo,:);   
currentG = reshape(currentG,numberNodes,numberNodes)'+eye(numberNodes);
% row stochastic for vector updates X(k+1) = X(k)*P
Weight_Row_Stoc = PRowStochastic(currentG,numberNodes); 
Weight_Doub_Stoc = PDoubleStochastic(currentG,numberNodes);
Laplacian_Mat = PLaplacian(currentG,numberNodes);
Diameter = 5;

% numGraphs = 100;
% numIniCond = 100;

%% Choose the objective function
index = 2; % index = 1 :Strongly Conv (Quadratic Prob), index = 2 :Conv(Least Squares) or huber loss, index = 3: Logistic Reg, index = 4: Huber loss 
convergence_Tol = 0.01;

%%% Quadratic Optimization
Dimension = 25; % , numberNodes  

%%% Least Squares
NumofFeatures = 15; % , numberNodes
Num_Examples = 1000;

%%% Logistic Regression
Training_Examples = 20;
Num_Features = 15;


vec_Dim = NumofFeatures;

%% Problem data initialization
[A, B, IniEstimate, C, H, h, Lh, Lfhat, mu, x_true] = Initialize_Test( numberNodes, vec_Dim, Num_Examples, Training_Examples, index );


%% CVX code to solve the problem centrally

Xstar = x_true.*ones(vec_Dim,numberNodes);
fstar = objective(A,B,C,Xstar,numberNodes,mu,index);


%% Distributed implementation
% stepsize = [0.5, 0.1, 0.05, 0.01];
Max_Iter = 50*Diameter;
acc = 0;
Iterdiff = zeros(3,Max_Iter);
for eta_index = 1 %1:3

gm = 1e1;

constrained  = 0;
unconstrained = 0;
multiloop = 1;


%% Comparison with constrained optimization algorithms on logistic regression 
if constrained == 1

[sol_res_DDistADMM, objval_err_DDistADMM, Iterdiff(eta_index,:), constraint_resk, t_DDistADMM, eq_constraint_resk, lam] = DDistADMM(vec_Dim, numberNodes,Weight_Doub_Stoc, A, B, ...
                                                                                                                                    IniEstimate, C, H, h, Diameter, ...
                                                                                                                                    gm, index, Xstar, fstar,Max_Iter, mu, Lh, eta_index,acc);
% % CD1 : ConstraintDist1
% [sol_res_CD1, objval_err_CD1, comm_CD1, constraint_resk_CD1, t_CD1, eq_constraint_resk_CD1] = ConstraintDist1(vec_Dim, numberNodes, Laplacian_Mat, A, B, ...
%                                                                                                               IniEstimate, C, H, h, index, Xstar, fstar,...
%                                                                                                               10*Max_Iter, mu, acc);       
% 
% % CD2 : ConstraintDist2
% [sol_res_CD2, objval_err_CD2, comm_CD2, constraint_resk_CD2, t_CD2, eq_constraint_resk_CD2] = ConstraintDist2(vec_Dim, numberNodes, Laplacian_Mat, A, B, ...
%                                                                                                               IniEstimate, C, H, h, index, Xstar, fstar,...
%                                                                                                               10*Max_Iter, mu, acc);
% Plotting 
Iterdiff1 = Iterdiff(eta_index,:); 
NumComm = cumsum(Iterdiff1);

figure(1); % objective function value comparison based on total iterations 
plot(NumComm, objval_err_DDistADMM(1:1:length(objval_err_DDistADMM)),'g', 'LineWidth', 2);
hold on 
plot(1:length(objval_err_CD1), objval_err_CD1(1:1:length(objval_err_CD1)),'r','LineWidth', 2);
hold on
plot(1:length(objval_err_CD2), objval_err_CD2(1:1:length(objval_err_CD2)),'b', 'LineWidth', 2);
l = legend('DC-DistADMM', 'CDA1', 'CDA2'); 
set(l,'interpreter','latex')
ylabel('$\mathbf{F}(\mathbf{\widehat{x}}) - \mathbf{F}(\mathbf{x}^*)$','interpreter','latex'); 
xlabel('Total iterations');

figure(2); % solution residual value comparison based on total iterations 
% plot(NumComm, sol_res_DDistADMM(1:1:length(sol_res_DDistADMM)),'g', 'LineWidth', 2);
% hold on 
plot(1:length(sol_res_CD1), sol_res_CD1(1:1:length(sol_res_CD1)),'r','LineWidth', 2);
hold on
plot(1:length(sol_res_CD2), sol_res_CD2(1:1:length(sol_res_CD2)),'b', 'LineWidth', 2);
% l = legend('DC-DistADMM', 'CDA1', 'CDA2'); 
l = legend('CDA1', 'CDA2');
set(l,'interpreter','latex')
ylabel('$\displaystyle \frac{\|\mathbf{x}^k - \mathbf{x}^*\|^2}{\|\mathbf{x}^0 - \mathbf{x}^*\|^2}$','interpreter','latex'); 
xlabel('Total iterations');

figure(3); % consensus constraint residual comparison based on total iterations 
% plot(NumComm, constraint_resk(1:1:length(constraint_resk)),'g', 'LineWidth', 2);
% hold on 
plot(1:length(constraint_resk_CD1), constraint_resk_CD1(1:1:length(constraint_resk_CD1)),'r','LineWidth', 2);
hold on
plot(1:length(constraint_resk_CD2), constraint_resk_CD2(1:1:length(constraint_resk_CD2)),'b', 'LineWidth', 2);
% l = legend('DC-DistADMM', 'CDA1', 'CDA2'); 
l = legend('CDA1', 'CDA2');
set(l,'interpreter','latex')
ylabel('Consensus Constraint','interpreter','latex'); 
xlabel('Total iterations');

figure(4); % equality constraint residual comparison based on total iterations 
% plot(NumComm, eq_constraint_resk(1:1:length(eq_constraint_resk)),'g', 'LineWidth', 2);
% hold on 
plot(1:length(eq_constraint_resk_CD1), eq_constraint_resk_CD1(1:1:length(eq_constraint_resk_CD1)),'r','LineWidth', 2);
hold on
plot(1:length(eq_constraint_resk_CD2), eq_constraint_resk_CD2(1:1:length(eq_constraint_resk_CD2)),'b', 'LineWidth', 2);
l = legend('CDA1', 'CDA2'); 
% l = legend('DC-DistADMM', 'CDA1', 'CDA2');
set(l,'interpreter','latex')
ylabel('$\|\mathbf{H}\widehat{\mathbf{x}}^k - \mathbf{h}\|$','interpreter','latex'); 
xlabel('Total iterations');        
                                                                      
figure(5); % solution residual value comparison based on CPU time
% plot(t_DDistADMM, sol_res_DDistADMM(1:1:length(sol_res_DDistADMM)),'g', 'LineWidth', 2);
% hold on
plot(cumsum(t_CD1), sol_res_CD1(1:1:length(sol_res_CD1)),'r', 'LineWidth', 2);
hold on
plot(cumsum(t_CD2), sol_res_CD2(1:1:length(sol_res_CD2)),'b', 'LineWidth', 2);
l = legend('CDA1', 'CDA2'); 
% l = legend('DC-DistADMM', 'CDA1', 'CDA2');
set(l,'interpreter','latex')
ylabel('$\displaystyle \frac{\|\mathbf{x}^k - \mathbf{x}^*\|^2}{\|\mathbf{x}^0 - \mathbf{x}^*\|^2}$','interpreter','latex'); 
xlabel('CPU Time(s)');
                                                                                                          
end
                                                                                                          
%% Comparison with Primal and dual unconstrained algorithms on Huber loss                                                                                                        
if unconstrained == 1

[sol_res_DDistADMM, objval_err_DDistADMM, Iterdiff(eta_index,:), constraint_resk, t_DDistADMM, eq_constraint_resk, lam] = DDistADMM(vec_Dim, numberNodes,Weight_Doub_Stoc, A, B, ...
                                                                                                                                    IniEstimate, C, H, h, Diameter, ...
                                                                                                                                    gm, index, Xstar, fstar,Max_Iter, mu, Lh, eta_index,acc);        
[sol_res_DGD, objval_err_DGD, comm_DGD, t_DGD] = DGD(numberNodes, Weight_Doub_Stoc, A, B, IniEstimate, C, ...
                                                      index, Xstar, fstar, 3*Max_Iter, mu,acc);
                                                   
[sol_res_EXTRA, objval_err_EXTRA, comm_EXTRA, t_EXTRA] = EXTRA(numberNodes, Weight_Doub_Stoc, A, B, IniEstimate, C, ...
                                                      index, Xstar, fstar,3*Max_Iter, mu,acc);
                                                   
[sol_res_PushPull, objval_err_PushPull, comm_PushPull, t_PushPull] = PushPull(numberNodes, Weight_Row_Stoc, A, B, IniEstimate, C, ...
                                                      index, Xstar, fstar, 3*Max_Iter, mu,acc);
 
% [sol_res_WShiADMM, objval_err_WShiADMM, comm_WShiADMM, t_WShiADMM] =  WShiADMM(numberNodes, currentG, A, B, IniEstimate, C,...
%                                                       index, Xstar, fstar, Max_Iter, mu,acc);                                                 

[sol_res_MultiAgentADMM, objval_err_MultiAgentADMM, comm_MultiAgentADMM, t_MultiAgentADMM] = MultiAgentADMM(numberNodes,...
                                                      currentG, A, B, IniEstimate, C,...
                                                      index, Xstar, fstar,3*Max_Iter,Lh,mu,acc);

% Plotting 
Iterdiff1 = Iterdiff(eta_index,:); 
NumComm = 0.3*cumsum(Iterdiff1);

figure(1); % objective function value comparison based on total iterations 
plot(NumComm, objval_err_DDistADMM(1:1:length(objval_err_DDistADMM)),'g', 'LineWidth', 2);
hold on 
plot(1:length(objval_err_DGD), objval_err_DGD(1:1:length(objval_err_DGD)), 'r', 'LineWidth', 2);
hold on
plot(1:length(objval_err_EXTRA), objval_err_EXTRA(1:1:length(objval_err_EXTRA)), 'b', 'LineWidth', 2);
hold on
plot(1:length(objval_err_PushPull), objval_err_PushPull(1:1:length(objval_err_PushPull)),'k', 'LineWidth', 2);
hold on
% plot(1:length(objval_err_WShiADMM), objval_err_WShiADMM(1:1:length(objval_err_WShiADMM)), 'm', 'LineWidth', 2);
% hold on
plot(1:length(objval_err_MultiAgentADMM), objval_err_MultiAgentADMM(1:1:length(objval_err_MultiAgentADMM)), 'w', 'LineWidth', 2);
hold on
% l = legend('DC-DistADMM', 'DGD', 'EXTRA', 'PushPull', 'DCOADMM', 'MultiAgentADMM'); 
l = legend('DC-DistADMM', 'DGD', 'EXTRA', 'PushPull', 'MultiAgentADMM');
set(l,'interpreter','latex')
ylabel('$\mathbf{F}(\mathbf{\widehat{x}}) - \mathbf{F}(\mathbf{x}^*)$','interpreter','latex'); 
xlabel('Total iterations');

figure(2); % solution residual value comparison based on total iterations 
plot(1:length(sol_res_DDistADMM), sol_res_DDistADMM(1:1:length(sol_res_DDistADMM)),'g', 'LineWidth', 2);
hold on
plot(NumComm, sol_res_DDistADMM(1:1:length(sol_res_DDistADMM)),'g', 'LineWidth', 2);
hold on
plot(1:length(sol_res_EXTRA), sol_res_EXTRA(1:1:length(sol_res_EXTRA)), 'b', 'LineWidth', 2);
hold on
plot(1:length(sol_res_PushPull), sol_res_PushPull(1:1:length(sol_res_PushPull)),'k', 'LineWidth', 2);
hold on
% plot(1:length(sol_res_WShiADMM), sol_res_WShiADMM(1:1:length(sol_res_WShiADMM)), 'm', 'LineWidth', 2);
% hold on
plot(1:length(sol_res_MultiAgentADMM), sol_res_MultiAgentADMM(1:1:length(sol_res_MultiAgentADMM)), 'w', 'LineWidth', 2);
hold on
plot(1:length(sol_res_DGD), sol_res_DGD(1:1:length(sol_res_DGD)), 'r', 'LineWidth', 2);
hold on
% l = legend('DC-DistADMM', 'DGD', 'EXTRA', 'PushPull', 'DCOADMM', 'MultiAgentADMM'); 
l = legend('DC-DistADMM','DC-DistADMM', 'EXTRA', 'PushPull', 'MultiAgentADMM','DGD');
set(l,'interpreter','latex')
ylabel('$\displaystyle \frac{\|\mathbf{x}^k - \mathbf{x}^*\|^2}{\|\mathbf{x}^0 - \mathbf{x}^*\|^2}$','interpreter','latex'); 
xlabel('Total iterations');

figure(3); % objective function value comparison based on CPU time 
plot(t_DDistADMM, objval_err_DDistADMM(1:1:length(objval_err_DDistADMM)),'g', 'LineWidth', 2);
hold on 
plot(t_DGD, objval_err_DGD(1:1:length(objval_err_DGD)), 'r', 'LineWidth', 2);
hold on
plot(t_EXTRA, objval_err_EXTRA(1:1:length(objval_err_EXTRA)), 'b', 'LineWidth', 2);
hold on
plot(t_PushPull, objval_err_PushPull(1:1:length(objval_err_PushPull)),'k', 'LineWidth', 2);
hold on
% plot(t_WShiADMM, objval_err_WShiADMM(1:1:length(objval_err_WShiADMM)), 'm', 'LineWidth', 2);
% hold on
plot(t_MultiAgentADMM, objval_err_MultiAgentADMM(1:1:length(objval_err_MultiAgentADMM)), 'w', 'LineWidth', 2);
hold on
% l = legend('DC-DistADMM', 'DGD', 'EXTRA', 'PushPull', 'DCOADMM', 'MultiAgentADMM'); 
l = legend('DC-DistADMM', 'DGD', 'EXTRA', 'PushPull', 'MultiAgentADMM');
set(l,'interpreter','latex')
ylabel('$\mathbf{F}(\mathbf{\widehat{x}}) - \mathbf{F}(\mathbf{x}^*)$','interpreter','latex'); 
xlabel('CPU Time(s)');

figure(4); % solution residual value comparison based on CPU time
plot(t_DDistADMM, sol_res_DDistADMM(1:1:length(sol_res_DDistADMM)),'g', 'LineWidth', 2);
hold on
plot(t_DGD, sol_res_DGD(1:1:length(sol_res_DGD)), 'r', 'LineWidth', 2);
hold on
plot(t_EXTRA, sol_res_EXTRA(1:1:length(sol_res_EXTRA)), 'b', 'LineWidth', 2);
hold on
plot(t_PushPull, sol_res_PushPull(1:1:length(sol_res_PushPull)),'k', 'LineWidth', 2);
hold on
% plot(t_WShiADMM, sol_res_WShiADMM(1:1:length(sol_res_WShiADMM)), 'm', 'LineWidth', 2);
% hold on
plot(t_MultiAgentADMM, sol_res_MultiAgentADMM(1:1:length(sol_res_MultiAgentADMM)), 'w', 'LineWidth', 2);
hold on
% l = legend('DC-DistADMM', 'DGD', 'EXTRA', 'PushPull', 'DCOADMM', 'MultiAgentADMM'); 
l = legend('DC-DistADMM', 'DGD', 'EXTRA', 'PushPull', 'MultiAgentADMM'); 
set(l,'interpreter','latex')
ylabel('$\displaystyle \frac{\|\mathbf{x}^k - \mathbf{x}^*\|^2}{\|\mathbf{x}^0 - \mathbf{x}^*\|^2}$','interpreter','latex'); 
xlabel('CPU Time(s)');


end

%% Comparison with multi-loop consensus algorithms on Huber loss

if multiloop == 1
    
[sol_res_DDistADMM, objval_err_DDistADMM, Iterdiff(eta_index,:), constraint_resk, t_DDistADMM, eq_constraint_resk, lam] = DDistADMM(vec_Dim, numberNodes,Weight_Doub_Stoc, A, B, ...
                                                                                                                                    IniEstimate, C, H, h, Diameter, ...
                                                                                                                                    gm, index, Xstar, fstar,Max_Iter, mu, Lh, eta_index,acc);  
                                                                                                                                
[sol_res_MouraFast, objval_err_MouraFast, comm_MouraFast, t_MouraFast] = MouraFast(vec_Dim, numberNodes, Weight_Doub_Stoc, A, B, IniEstimate, C, ...
                                                                                   index, Xstar, fstar,Max_Iter, mu, acc);

                                                                           
[sol_res_nearDGD, objval_err_nearDGD, comm_nearDGD, t_nearDGD] = nearDGD(vec_Dim, numberNodes, Weight_Doub_Stoc, A, B, IniEstimate, C,...
                                                                          index, Xstar, fstar,Max_Iter, mu, acc);                                                                           
                                                                           

% Plotting
Iterdiff1 = Iterdiff(eta_index,:); 
NumComm = cumsum(Iterdiff1);

% figure(1); % objective function value comparison based on total iterations 
% plot(NumComm, objval_err_DDistADMM(1:1:length(objval_err_DDistADMM)),'g', 'LineWidth', 2);
% hold on 
% plot(cumsum(comm_MouraFast), objval_err_MouraFast(1:1:length(objval_err_MouraFast)), 'r', 'LineWidth', 2);
% hold on
% plot(cumsum(comm_nearDGD), objval_err_nearDGD(1:1:length(objval_err_nearDGD)-1), 'b', 'LineWidth', 2);
% l = legend('DC-DistADMM', 'FDGD', 'nearDGD'); 
% set(l,'interpreter','latex')
% ylabel('$\mathbf{F}(\mathbf{\widehat{x}}) - \mathbf{F}(\mathbf{x}^*)$','interpreter','latex'); 
% xlabel('Total iterations');

figure(2); % solution residual value comparison based on total iterations 
plot([0,NumComm(1:1:length(NumComm) - 1)], sol_res_DDistADMM(1:1:length(sol_res_DDistADMM)),'g', 'LineWidth', 2);
hold on
plot(cumsum(comm_MouraFast), sol_res_MouraFast(1:1:length(sol_res_MouraFast)-1), 'r', 'LineWidth', 2);
hold on
plot(cumsum(comm_nearDGD), sol_res_nearDGD(1:1:length(sol_res_nearDGD)-1), 'b', 'LineWidth', 2);
l = legend('DC-DistADMM', 'FDGD', 'nearDGD'); 
set(l,'interpreter','latex')
ylabel('$\displaystyle \frac{\|\mathbf{x}^k - \mathbf{x}^*\|^2}{\|\mathbf{x}^0 - \mathbf{x}^*\|^2}$','interpreter','latex'); 
xlabel('Total iterations');

% figure(3); % objective function value comparison based on CPU time 
% plot(t_DDistADMM, objval_err_DDistADMM(1:1:length(objval_err_DDistADMM)),'g', 'LineWidth', 2);
% hold on 
% plot(t_MouraFast, objval_err_MouraFast(1:1:length(objval_err_MouraFast)), 'r', 'LineWidth', 2);
% hold on
% plot(t_nearDGD, objval_err_nearDGD(1:1:length(objval_err_nearDGD)), 'b', 'LineWidth', 2);
% l = legend('DC-DistADMM', 'FDGD', 'nearDGD'); 
% set(l,'interpreter','latex')
% ylabel('$\mathbf{F}(\mathbf{\widehat{x}}) - \mathbf{F}(\mathbf{x}^*)$','interpreter','latex'); 
% xlabel('CPU Time(s)');
CummTime = cumsum(t_DDistADMM);
figure(4); % solution residual value comparison based on CPU time
plot([0,CummTime(1:1:length(CummTime)-1)], sol_res_DDistADMM(1:1:length(sol_res_DDistADMM)),'g', 'LineWidth', 2);
hold on
plot(cumsum(t_MouraFast), sol_res_MouraFast(1:1:length(sol_res_MouraFast)-1), 'r', 'LineWidth', 2);
hold on
plot(cumsum(t_nearDGD), sol_res_nearDGD(1:1:length(sol_res_nearDGD)-1), 'b', 'LineWidth', 2);
l = legend('DC-DistADMM', 'FDGD', 'nearDGD'); 
set(l,'interpreter','latex')
ylabel('$\displaystyle \frac{\|\mathbf{x}^k - \mathbf{x}^*\|^2}{\|\mathbf{x}^0 - \mathbf{x}^*\|^2}$','interpreter','latex'); 
xlabel('CPU Time(s)');

Commsteps = Iterdiff(eta_index,:);
figure(5); % communication cost comparison
plot(1:length(Commsteps), Commsteps(1:1:length(Commsteps)),'g','LineWidth',2);
hold on
plot(1:length(comm_MouraFast), comm_MouraFast(1:1:length(comm_MouraFast)),'r','LineWidth',2);
hold on 
plot(1:length(comm_nearDGD), comm_nearDGD(1:1:length(comm_nearDGD)),'b','LineWidth',2);
l = legend('DC-DistADMM', 'FDGD', 'nearDGD'); 
set(l,'interpreter','latex')
ylabel('# communication steps'); 
xlabel('Algorithm iterations');
                                                                  
end

end