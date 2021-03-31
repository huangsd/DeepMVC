function [Finres, Hres, result, obj] = DeepMVC(X, layers, labels, maxiter)
%
%solve the following problem£º
% Deep MF for learning multi-view data representation
% min\sum_m\alpha^{(m)}||X^{(m)}-U_1^{(m)}U_2^{(m)}...U_r^{(m)}V_r^{T}||_{2,1}
% s.t. V_i>0, i\in [1, ..., r-1], V_r = {0,1}.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% written by Shudong Huang on 15/1/2019
% Reference: Shudong Huang, Zhao Kang, Zenglin Xu. 
% Auto-weighted multi-view clustering via deep matrix decomposition. 
% In: Pattern Recognition.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% ATTN1: This package is free for academic usage. The code was developed by Mr. SD. Huang (huangsd@std.uestc.edu.cn). You can run
% it at your own risk. For other purposes, please contact Prof. Zenglin Xu (zenglin@gmail.com)
%
% ATTN2: This package was developed by Mr. SD. Huang (huangsd@std.uestc.edu.cn). For any problem concerning the code, please feel
% free to contact Mr. Huang.
%
% Input:
% X: cell array, view_num x 1, each array is dim x num
% labels: groundtruth, num x 1 
% k: num of clusters
% H: num x k
% Z: dim x k
% rho: penalty parameter
% ratio: the ratio of outliers, set it according to the paper
% e.g., layers = [100 50] ;
%
% Output:
% Hres: the clustering indicator matrix
% obj: objective value
% 
% Example:
% [Finres, Hres, result3, obj] = DeepMVC(data', [100 50 length(unique(labels))], labels, 50);
%
disp('Deep multi-view learning');
thresh=1e-7;
% number of views
view_num = max(size(X));
% number of clusters
C_num = length(unique(labels));
% number of layers
layer_num = numel(layers);
% dimension of data
 [mFea, nSmp] = size(X{1});

% =====================   Normalization =====================
% for i = 1:view_num
%     for  j = 1:nSmp
%         X{i}(:,j) = (X{i}(:,j) - mean( X{i}(:,j))) / std(X{i}(:,j)) ;
%     end
% end

% initialize each layer for each view
Z = cell(view_num, layer_num);
H = cell(view_num, layer_num);
for i_view = 1:view_num
    for i_layer = 1:layer_num
        if i_layer == 1
           V = X{i_view}; 
       % For the first layer we go linear from X to Z*H
           else
           V = H{i_view,i_layer-1};
        end
        if i_layer == 1
           disp('The 1st layer ...');
           else if i_layer == 2
             disp('The 2nd layer ...');
           else
             fprintf('the %d-th layer\n', i_layer);
           end
        end
        if i_layer > 1
           V = V';
        end 
        % initialize input with kmeans
          [Z{i_view,i_layer}, H{i_view,i_layer}] = KMeansdata(V, layers(i_layer));
        
        % initialize input with NMF
        %  [Z{i_view,i_layer}, H{i_view,i_layer}] = NMFdata(V, layers(i_layer));
    end
end

% normalization
% X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));

% initialize H_ and \alpha
H_err = cell(view_num, layer_num);
alpha = (1/view_num)*ones(view_num,1);

% initialize wieght
for i_view = 1:view_num  
    DD{i_view} = diag(ones(nSmp, 1))*alpha(i_view);
end

% update ...
for iter = 1:maxiter  
    for i_view = 1:view_num
        H_err{i_view,numel(layers)} = H{i_view,numel(layers)};
    end
    % compute H_
    for i_view = 1:view_num
        for i_layer = numel(layers)-1:-1:1
            H_err{i_view,i_layer} = (Z{i_view,i_layer+1} * H_err{i_view,i_layer+1}')';
        end
    end
    
% compute D_m
%         for i_view = 1:view_num
%              D{i_view} = computeD(X{i_view}, Z{i_view,1}, H_err{i_view,1});
%              DD{i_view} = alpha(i_view)*D{i_view};
%         end
   
    for i_view = 1:view_num
        for i_layer = 1:numel(layers)   
            % update Z, i.e., U
            if i_layer == 1           
               Z{i_view,1} = (X{i_view}*DD{i_view}*H_err{i_view,1})/(H_err{i_view,1}'*DD{i_view}*H_err{i_view,1});  %  pinv    
            else
           
               Z{i_view,i_layer} = (ZZ'*ZZ)\(ZZ'*X{i_view}*DD{i_view}*H_err{i_view,i_layer})/(H_err{i_view,i_layer}'*DD{i_view}*H_err{i_view,i_layer});
           
            end
        
            if i_layer == 1
               ZZ = Z{i_view,1};
            else
               ZZ = ZZ*Z{i_view,i_layer};
            end
            ZZ1{i_view} = ZZ;

            % update H_i, i.e., V_i (i<r)
            
            if i_layer < numel(layers)
               % H{view_num,i_layer} = (X{i_view}'*ZZ)*pinv(ZZ'*ZZ);
               H{view_num,i_layer} = (X{i_view}'*ZZ)/(ZZ'*ZZ);  
            end             
        end
    end
    
    % update H_r, i.e., V_i (i=r)  
    for ni = 1:nSmp
        dVec = zeros(view_num, 1);
        for i_view = 1:view_num
            xVec{i_view} = X{i_view}(:,ni);
            tt = diag(DD{i_view});
            dVec(i_view, 1) = tt(ni);
         end
         HH(ni,:) = searchBestIndicator(dVec, xVec, ZZ1);
    end
     
    for i_view = 1:view_num
        H{i_view,numel(layers)} = HH;
    end
     
    % update the weight for each view
    for i_view = 1:view_num
        XX{i_view} = (X{i_view}-Z{i_view,1}*H_err{i_view,1}')';
        XX2{i_view} = sqrt(sum(XX{i_view}.*XX{i_view},2));%||X-UV'||_{2}
        temp(i_view) = sum(XX2{i_view}); %||X-UV'||_{2,1}
        alpha(i_view) = 0.5/sqrt(temp(i_view));
    end
%     while (max(alpha)<0.1)
%         alpha = 10*alpha;
%     end

        % update D
         for i_view = 1:view_num             
             XX{i_view} = (X{i_view}-Z{i_view,1}*H_err{i_view,1}')';
             XX2{i_view} = sqrt(sum(XX{i_view}.*XX{i_view},2)+eps);  
             DD{i_view} = sparse(diag(0.5./XX2{i_view}*alpha(i_view)));
         end

    % calculate the objective
    obj(iter) = 0;
    for i_view = 1:view_num
        obj(iter) = obj(iter) + (alpha(i_view))*sum(XX2{i_view});
    end
    if(iter > 1)
        diff = abs(obj(iter-1) - obj(iter))/obj(iter-1);
        if(diff < thresh)
            break;
        end
    end
    
end
    
% 
Hres = H{numel(H)};
% Finres = litekmeans(H',nClass,'Replicates',100);
% Finres = litekmeans(Hres,length(unique(labels)),'Replicates',20);
for i = 1:size(Hres,2)
    Hres(:,i) = i*Hres(:,i);
end
Finres = sum(Hres,2);

% result = [ACC MIhat Purity];
result = ClusteringMeasure(labels, Finres)  

end

% function searchBestIndicator
function outVec = searchBestIndicator(dVec, xCell, F)
% solve the following problem,
numView = length(F);
c = size(F{1}, 2);
tmp = eye(c);
obj = zeros(c, 1);
for j = 1: c
    for v = 1: numView
        obj(j,1) = obj(j,1) + dVec(v) * (norm(xCell{v} - F{v}(:,j))^2);
    end
end
[min_val, min_idx] = min(obj);
outVec = tmp(:, min_idx);
end