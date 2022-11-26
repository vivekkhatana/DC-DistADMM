function [sol_res, objval_res, comm, constraint_res, ftime, eq_constraint_res, lam] = DDistADMM(vec_Dim, numberNodes,...
                                                      Weight_Mat, A, B, IniEstimate, C, H, h, Diameter,...
                                                      gm, index, Xstar, fstar, Max_Iter, mu, Lh, k, acc)
        

               
        sol_res = []; 
        objval_res = [];       
        constraint_res = [];
        eq_constraint_res = [];
        lam = [];
        X = [];
        X_hat = [];
        Y = []; 
        Y_hat =[];
        Lambda = [];
        Mu = [];
        Xi = [];
        X(:,:,1)= IniEstimate;
        Y(:,:,1)= IniEstimate;
        Lambda(:,:,1)= zeros(vec_Dim,numberNodes);
        [t1, t2] = size(H);
        Mu(:,:,1)= zeros(t1,numberNodes);
        Xi(:,:,1) = zeros(1000,100);
        X_hat(:,:,1)= IniEstimate;
        Y_hat(:,:,1)= IniEstimate;
        
%         Iterdiff = [];
        comm = [];
        itime = cputime;
 
        for i = 1:1:Max_Iter
            
               if index == 1
                   
                   for j = 1:numberNodes
                      X(:,j,i+1) = (A{j} + gm*eye(vec_Dim))\( gm*Y(:,j,i) - Lambda(:,j,i) - B(:,j) );
                   end
                   
               elseif index ==2
                   
                   for j = 1:numberNodes
                      X(:,j,i+1) = (A{j}'*A{j} + gm*eye(vec_Dim))\( gm*Y(:,j,i) - Lambda(:,j,i) + A{j}'*B(:,j) );
                   end
                   
                   
               elseif index == 3
               % Here we used FISTA followed by a projection step
               
                   s = [];
                   s(:,:,1) = X(:,:,i);
                   z = [];
                   z(:,:,2) = X(:,:,i);
                   alpha = [];
                   alpha(1) = 1;
                   alpha(2) = (1+sqrt(5))/2;
                   
                   
               for t = 2:5  
                   alpha(t+1) = (1+sqrt(4*(alpha(t))^2+1))/2;
                   grad = gradient( A, B, z(:,:,t), numberNodes, mu, index);
                   for j = 1:numberNodes
%                     var1 = (1/(Lh + gm))*(Lh*z(:,j,t) + gm*Y(:,j,i) - grad(:,j) - Lambda(:,j,i));
                    var1 = z(:,j,t) - (1/Lh)*( gm*(eye(vec_Dim)+H'*H)*z(:,j,t) + grad(:,j) ...
                                              - gm*Y(:,j,i) - gm*H'*h + Lambda(:,j,i) + H'*Mu(:,j,i) );
                    var2 = mu/(Lh)*ones(length(z(:,j,t)),1);
                    s(:,j,t) = shrinkage(var1, var2);
                  
                       if norm(s(:,j,t),2) <= 1
                           s(:,j,t) = s(:,j,t);
                       else
                           s(:,j,t) = s(:,j,t)/norm(s(:,j,t),2);
                       end 
                       
                   end
                   z(:,:,t+1) = s(:,:,t) + ((alpha(t)-1)/alpha(t+1))*(s(:,:,t) - s(:,:,t-1)); 
               end                 
               
               X(:,:,i+1) = s(:,:,5);
               
               elseif index==4
                   % Here we used FISTA
               
                   s = [];
                   s(:,:,1) = X(:,:,i);
                   z = [];
                   z(:,:,2) = X(:,:,i);
                   alpha = [];
                   alpha(1) = 1;
                   alpha(2) = (1+sqrt(5))/2;
                   
                   
               for t = 2:5  
                   alpha(t+1) = (1+sqrt(4*(alpha(t))^2+1))/2;
                   grad = gradient( A, B, z(:,:,t), numberNodes, mu, index);
                   for j = 1:numberNodes
%                     var1 = (1/(Lh + gm))*(Lh*z(:,j,t) + gm*Y(:,j,i) - grad(:,j) - Lambda(:,j,i));
                    var1 = z(:,j,t) - (1/Lh)*( gm*z(:,j,t) + grad(:,j) ...
                                              - gm*Y(:,j,i) + Lambda(:,j,i) );
                    var2 = (3/Lh)*ones(length(z(:,j,t)),1);
                    s(:,j,t) = shrinkage(var1, var2);
                       
                   end
                   z(:,:,t+1) = s(:,:,t) + ((alpha(t)-1)/alpha(t+1))*(s(:,:,t) - s(:,:,t-1)); 
               end                 
               
               X(:,:,i+1) = s(:,:,5);
                   
                   
               end
               
               if k == 1
                   rho = 0.01/i^(1.5);
               elseif k == 2
                   rho = 1/i^(3);
               elseif k == 3
                   rho = (0.85)^i;
               end
               
               U = X(:,:,i+1) + (1/gm)*Lambda(:,:,i);
               [Y(:,:,i+1),Iterdiff] = Consensus(Weight_Mat,Diameter,U,rho);
               
               comm(i) = Iterdiff + 1;
               
               Lambda(:,:,i+1) = Lambda(:,:,i) + gm*(X(:,:,i+1) - Y(:,:,i+1));
               
               lam = [lam Lambda(:,:,i+1)];
               
               if index == 3
                   for j = 1:numberNodes
                        Mu(:,j,i+1) = Mu(:,j,i) + gm*(H*X(:,j,i+1) -  h);
                   end
               end

               X_hat(:,:,i+1) = 0.5*(X_hat(:,:,i) + X(:,:,i+1));
               Y_hat(:,:,i+1) = 0.5*(Y_hat(:,:,i) + Y(:,:,i+1));
   
               var_sol = X(:,:,i) - Xstar;
               var_inisol = IniEstimate - Xstar;
                              
               if k==1
                   sol_res(i) = (1/i^0.35)*norm(var_sol,'fro')^2/norm(var_inisol,'fro')^2;
                   objval_res(i) = (1/i^0.35)*objective(A,B,C,X_hat(:,:,i+1),numberNodes,mu,index);
                   constraint_res(i) =  (1/i^0.35)*norm((X_hat(:,:,i+1) - Y_hat(:,:,i+1)),'fro');
                   
                   if index == 3
                       eq_cons = 0;
                       for j = 1:numberNodes
                         eq_cons = eq_cons + norm(H*X_hat(:,j,i+1) -  h, 'fro');
                       end
                       eq_constraint_res(i) = (1/i^0.35)*eq_cons;
                   end
                      
               elseif k==2
                   sol_res(i) = (1/i^0.55)*norm(var_sol,'fro')^2/norm(var_inisol,'fro')^2;
                   objval_res(i) = (1/i^0.55)*objective(A,B,C,X_hat(:,:,i+1),numberNodes,mu,index);
                   constraint_res(i) =  (1/i^0.55)*norm((X_hat(:,:,i+1) - Y_hat(:,:,i+1)),'fro');
                   
                   if index == 3
                       eq_cons = 0;
                       for j = 1:numberNodes
                         eq_cons = eq_cons + norm( H*X_hat(:,j,i+1) -  h, 'fro');
                       end
                       eq_constraint_res(i) = (1/i^0.55)*eq_cons;
                   end
                   
               elseif k==3
                   sol_res(i) = (1/i)*norm(var_sol,'fro')^2/norm(var_inisol,'fro')^2;
                   objval_res(i) = (1/i)*objective(A,B,C,X_hat(:,:,i+1),numberNodes,mu,index);
                   constraint_res(i) = (1/i)*norm((X_hat(:,:,i+1) - Y_hat(:,:,i+1)),'fro');
                   
                   if index == 3
                       eq_cons = 0;
                       for j = 1:numberNodes
                         eq_cons = eq_cons + norm( H*X_hat(:,j,i+1) -  h, 'fro');
                       end
                       eq_constraint_res(i) = (1/i)*eq_cons;
                   end
                   
               end

              ftime(i) = cputime - itime; 
              if acc ~= 0
                  if sol_res(i) <= acc
                       break;
                    else 
                        comm = 0;
                  end
              end
                 
        end                
end 