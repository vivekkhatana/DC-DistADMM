% This code is written to do dstributed optimization simulations

clear all

% load('graphArray_20Nodes_10000ConnectedDiGraph');
load('graphArray_10Nodes_100ConnectedDiGraph');
Num_graphs = size(arr,1);
numberNodes = sqrt(size(arr,2));
numQuad = 7;
numLin = 3;
% numGraphs = 100;
% numIniCond = 100;

Dimension = 10;
mean = 1;

% s = rng;
rng('default');
B = normrnd(0,1,[Dimension,numLin]);
rng('default');
Q = randn(Dimension,Dimension);
L = normrnd(0,1,[Dimension,numQuad]);
% P = Q' * diag(abs(mean+L(:,j))) * Q;

% save('s');

xstar = [0.0266, -0.0557, 0.1479,-0.0323,-0.0046,0.0135,-0.0872,0.0445,-0.1849,0.0046]';
Xstar = xstar.*ones(numberNodes,numberNodes);
rho = 2*[0.1, 0.001, 0.00001];
step_size = [5e-3, 1e-3, 5e-4, 2e-4, 1e-4];
% rho = 0.01*(1:5);
for iniCond = 1%:numIniCond
        X0 = rand(Dimension,numberNodes);
        Diam = 9;
        numIterations= 1000*Diam;
                        
        X = zeros(Dimension,numberNodes,numIterations+1);
        Y = zeros(Dimension,numberNodes,numIterations+1);
        Z = zeros(Dimension,numberNodes,numIterations+1);
        X(:,:,1)= X0;
        Y(:,:,1)= X0;
        Z(:,:,1)= X0;
        grad = zeros(Dimension,numberNodes,numIterations+1);
        Iterdiff = zeros(numIterations,1);

        R1 = zeros(numberNodes,numberNodes);
        R2 = zeros(numberNodes,numberNodes);
        Res = zeros(numberNodes,numIterations);
        r1 = zeros(numberNodes,numberNodes);
        r2 = zeros(numberNodes,numberNodes);
        res = zeros(numberNodes,numIterations);

  for rhok = 1   
    for alphak = 1
        for graphNo = 27

           currentG = arr(graphNo,:);   
           currentG = reshape(currentG,numberNodes,numberNodes)'+eye(numberNodes);
           Weight_Row_Stoc = PRowStochastic(currentG,numberNodes);
           alpha = step_size(alphak);
           
           for i = 2:2:numIterations
               for j = 1:numQuad
                  X(:,j,i) = X(:,j,i-1) - alpha*(Q'*diag(abs(mean+L(:,j)))*Q)*X(:,j,i-1);
               end
               for j = 8:7+numLin
                  X(:,j,i) = X(:,j,i-1) - alpha*(B(:,numberNodes-j+1));
               end
               [Z(:,:,i),Iterdiff(i)] = Consensus(Weight_Row_Stoc,numberNodes,Diam,X(:,:,i),rho(rhok));
               R1 = X(:,:,i)-Xstar;
               R2 = X0-Xstar;
               Res(:,i) = norm(R1(:,1))/norm(R2(:,1));               
               X(:,:,i+1) = Z(:,:,i); 
           end
            hh=zeros(1,floor(numIterations/2));
            hh(1)=1;
            for i = 2:2:numIterations
                hh(i/2+1) = hh(i/2)+i/2+Iterdiff(i);
            end
%             
%      for alpha1k =5      
%            alpha1= step_size(alpha1k);
%            numIterations1=(numIterations)*120;
%            for i = 1:numIterations1
%                for j = 1:numQuad
%                   grad(:,j,i) = alpha1*((Q'*diag(abs(mean+L(:,j)))*Q)*Y(:,j,i));
%                end
%                for j = 8:7+numLin
%                   grad(:,j,i) = alpha1*(B(:,numberNodes-j+1));
%                end
%                Y(:,:,i+1) = Y(:,:,i)-grad(:,:,i);
%                r1 = Y(:,:,i)-Xstar;
%                r2 = X0-Xstar;
%                res(:,i) = norm(r1(:,1))/norm(r2(:,1)); 
%            end
%      end
              
                             
        end
    end
  end
end
                             




ff = Res(10,2:2:numIterations);
plot(2:2:numIterations,ff);
% stairs(hh(1:end-1),ff);
% plot(2:2:numIterations,ff);
% gg = res(10,2:2:numIterations);
% plot(2:2:numIterations,gg);
% plot(2:2:numIterations,gg);

% legend('Our Scheme','Nedic')

% hold on;


% G=digraph(currentG);
% figure;           
% G1 = digraph(currentG,'omitselfloops');
% p=plot(G1,'LineWidth',5,'ArrowSize',16,'MarkerSize',20,'NodeColor',[.65 .65 .65], 'EdgeColor','m');grid on;set(gca,'XTick',[],'YTick',[]);
% nl = p.NodeLabel;
% p.NodeLabel = '';
% xd = get(p, 'XData');
% yd = get(p, 'YData');
% text(xd, yd, nl, 'FontSize',14, 'FontWeight','bold', 'HorizontalAlignment','center', 'VerticalAlignment','middle');
% clear p;



