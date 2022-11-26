function [xstar, fstar] = Centralized_CVX(vec_Dim, A, B, C, mu, Num_Agents, objindex,x_true)



   if objindex == 1  
       cvx_solver('seDuMi');
       cvx_begin quiet
       variables x(vec_Dim)
       objective_func = 0;
        for i = 1:Num_Agents
            objective_func = objective_func + 0.5*x'*A{i}*x + B(:,i)'*x + C(i);
        end
        minimize objective_func
        cvx_end
        xstar = x.*ones(vec_Dim,Num_Agents);
        fstar = objective(A,B,C,xstar,Num_Agents,mu,objindex);
        
   elseif objindex == 2
       cvx_solver('seDuMi');
       cvx_begin quiet
       variables x(vec_Dim)
       objective_func = 0;
       for i = 1:Num_Agents
        objective_func = objective_func + 0.5*power(2,norm(A{i}*x - B(:,i),2));
       end
        minimize objective_func
        cvx_end
        xstar = x.*ones(vec_Dim,Num_Agents);
        fstar = objective(A,B,C,xstar,Num_Agents,mu,objindex);
        
   elseif objindex == 3
%        cvx_solver('seDuMi');
%        cvx_begin quiet
%        variables z(vec_Dim+1,1)
%        objective_func = 0;
% %        for i = 1:Num_Agents
% %            [~,sample_size] = size(A{i});
% %            innersum(i) = sum(log(1 + exp(-A{i}*z(2:end) - B(:,i)*z(1)))); %+ sample_size*mu*norm(z(2:end),1));  
% %        end
% %        objective_func = sum(innersum);
%         minimize objective_func
%         cvx_end
        xstar = x_true.*ones(vec_Dim,Num_Agents);
        fstar = objective(A,B,C,xstar,Num_Agents,mu,objindex);
   end
   

end
