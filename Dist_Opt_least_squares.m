% This code is written to do distributed optimization simulations

clear all

% load('graphArray_20Nodes_10000ConnectedDiGraph');
load('graphArray_10Nodes_100ConnectedDiGraph');
Num_graphs = size(arr,1);
numberNodes = sqrt(size(arr,2));

% numGraphs = 100;
% numIniCond = 100;

Dimension = 10;

% s = rng;
rng('default');
B = normrnd(0,1,[Dimension,numberNodes]);
rng('default');
A = normrnd(0,1,[Dimension,Dimension,numberNodes]);
% save('s');

xstar = [0.3920,0.1139,-0.3212,0.0373,-0.0615,-0.0294,-0.1725,-0.1359,0.0639,0.0813]';
Xstar = xstar.*ones(numberNodes,numberNodes);
rho = 2*[0.1, 0.01, 0.001, 0.0001, 0.00001];
% rho = 0.01*(1:5);
for iniCond = 1%:numIniCond
        X0 = rand(Dimension,numberNodes);
        Diam = 9;
        numIterations= 100*Diam;
        alpha = 5e-3;
                
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

  for rhok = 1:5           
        for graphNo = 27

           currentG = arr(graphNo,:);   
           currentG = reshape(currentG,numberNodes,numberNodes)'+eye(numberNodes);
           Weight_Row_Stoc = PRowStochastic(currentG,numberNodes);
        
           for i = 2:2:numIterations
               for j = 1:numberNodes
                  X(:,j,i) = X(:,j,i-1) - alpha*(A(:,:,j)'*A(:,:,j)*X(:,j,i-1) - A(:,:,j)'*B(:,j));
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
           
%            alpha1=1e-4;
%            numIterations1=(numIterations)*120;
%            for i = 1:numIterations1
%                for j = 1:numberNodes
%                   grad(:,j,i) = alpha1*(A(:,:,j)'*A(:,:,j)*Y(:,j,i) - A(:,:,j)'*B(:,j));
%                end
%                Y(:,:,i+1) = Y(:,:,i)-grad(:,:,i);
%                r1 = Y(:,:,i)-Xstar;
%                r2 = X0-Xstar;
%                res(:,i) = norm(r1(:,1))/norm(r2(:,1)); 
%            end
               

       % hh = downsample(newRes(10,22:2:length(newRes)),12);
            ff = Res(10,2:2:numIterations);
            % ff = downsample(Res(10,2:2:numIterations),12);
            plot(2:2:numIterations,ff);
            % gg = res(10,1:1:numIterations);
            % plot((22:24:length(newRes)),hh);
            % stairs(hh(1:end-1),ff);
            hold on;
                             
        end
  end
                               
end


% hh = downsample(newRes(10,22:2:length(newRes)),12);
% ff = Res(10,2:2:numIterations);
% ff = downsample(Res(10,2:2:numIterations),12);
% plot(2:2:numIterations,ff);
% gg = res(10,1:1:numIterations);
% plot((22:24:length(newRes)),hh);
% stairs(hh(1:end-1),ff);
% hold on;

% plot(1:1:numIterations1,res(10,1:1:numIterations1));
% legend('Our Scheme','Nedic')


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



