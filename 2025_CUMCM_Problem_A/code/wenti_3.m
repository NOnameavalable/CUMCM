%问题3：假设FY1，FY2，FY3投掷一枚烟雾干扰弹，提出对M1的最优干扰策略
clear;clc;
%本题中使用的参数设置
% 采样点的数目
cap_1=64;
cap_2=8;
side_1=64;
side_2=12;

% 使用的时间步长
dt_kuai=0.20;   % 快速搜索阶段
dt_jing=0.02;   % 精准搜索阶段
t_edge=1e-3;   % 间隔时间
max_b_it=40;

% 定义度量安全数值
safety_ep=0.10;  

% 按照题目要求进行PSO-DE 的配置
n_p=120;   %设置粒子群的数模
dim=8;      %一共有8个维度
max_it=240;                   %最大迭代次数为240
eliteRt =0.30;  %精英率
Fm=0.6;         %交叉概率
CR=0.7;
w_min=0.4;
w_max=0.9;      %初始和结束时的结束惯性
c1=1.8;        %个体与社会的学习因子
c2=1.8;
i_e=25;      %每25代进行一次移民
i_rt=0.20;        %移民率为20%
c_f_cishu=50;                        %重复计算50次

%定义所需要使用的参数值
g=9.81;      % 定义重力加速度
R_yun=10;        %云团半径为10m
vd=3;       %云团下沉速度为3m/s
v_dao=300;                % 设导弹速度为300m/s
M1=[20000,0,2000];  % 导弹初始位置
uM=(-M1)/norm(M1); % 导弹M1指向原点的单位方向
t_end=norm(M1)/v_dao;     % 时间的上限，即导弹M1到原点所需的时间
FY1=[17800,0,1800];   % 无人机FY1的初始位置
% 设置圆柱体的参数
r_z=10;          
r_R=7;                
r_xy=[0,200,0]; 
    

%建立评估网格
tG_coa=0:dt_kuai:t_end;
P_coa=sampleCylinderPointsHP_(r_xy,r_R,r_z,24,6,24,4); % 引用辅助函数，一共246个点

tGrid_fine=0:dt_jing:t_end;
P_fine=sampleCylinderPointsHP_(r_xy,r_R,r_z,side_1,side_2,cap_1,cap_2); % 再次引用辅助函数，生成高密度网格

%目标函数
deco=@(x) decodeVars_with_gap_(x);
fit_coa=@(x) objective_cover_time_vec_( ...
    x,deco,M1,v_dao,uM,FY1,R_yun,vd,g, ...
    tG_coa,P_coa,safety_ep);

%核心部分：采用PSO-DE+重启循环 
best_all=struct('f',-inf,'x',[],'S',[],'drop_pts',[],'blow_pts',[], ...
    'D_B_G',[],'cover_Mask',[],'intervals',[],'totalCover',[]);

rng('shuffle');
for rs=1:c_f_cishu
    % 初始化粒子
    X=rand(n_p, dim);
    V=0.2*(rand(n_p,dim)-0.5);
    p_best_X=X;
    p_best_f=-inf(n_p,1);
    g_best_X=zeros(1,dim); g_best_F=-inf;

    % 进行初评估
    for i=1:n_p
        f=fit_coa(X(i,:));
        p_best_f(i)=f;
        p_best_X(i,:)=X(i,:);
        if f > g_best_F
               g_best_F=f;
               g_best_X=X(i,:);
        end
    end

    % 进行迭代运算
    for it=1:max_it
        w=w_max-(w_max-w_min)*(it-1)/(max_it-1); %惯性权重按线性递减
        % 进行PSO步骤
        for i=1:n_p
            r1=rand(1,dim); r2=rand(1,dim);
            V(i,:)=w*V(i,:)+c1*r1.*(p_best_X(i,:)-X(i,:))+c2*r2.*(g_best_X-X(i,:));
            V(i,:)=max(min(V(i,:),0.5),-0.5);
            X(i,:)=max(min(X(i,:)+V(i,:),1),0);
            f=fit_coa(X(i,:));
            if f>p_best_f(i)
                p_best_f(i)=f;
                p_best_X(i,:)=X(i,:);
                if f>g_best_F
                       g_best_F=f;
                       g_best_X=X(i,:);
                end
            end
        end
        % 有关精英的部分
        [~, idx]=sort(p_best_f,'descend');
        nElite=max(3, round(eliteRt*n_p));
        el_x=idx(1:nElite);
        for e=el_x'
            r=randperm(n_p,2);
            while any(r==e)
                   r=randperm(n_p,2);
            end
            xr1=X(r(1),:);
            xr2=X(r(2),:);
            x=X(e,:);
            v_D_E=x+Fm*(g_best_X-x)+Fm*(xr1-xr2);
            jrand=randi(dim);
            u=x;
            for j=1:dim
                 if rand<=CR || j==jrand
                        u(j)=v_D_E(j);
                end
            end
            u=max(min(u,1),0);
            fu=fit_coa(u);
            if fu>p_best_f(e)   %都属于跟新个人最优与全局最优
                X(e,:)=u;
                p_best_X(e,:)=u;
                p_best_f(e)=fu;
                if fu>g_best_F
                       g_best_F=fu;
                       g_best_X=u;
                 end
            end
        end
        % 移民部分，最差的20%移除
        if mod(it, i_e)==0
            [~, idxAsc] = sort(p_best_f,'ascend');
            nImm=max(1, round(i_rt*n_p));
            badIdx=idxAsc(1:nImm);
            X(badIdx,:)=rand(nImm, dim);
            V(badIdx,:)=0.2*(rand(nImm,dim)-0.5);
            for b=badIdx'
                f=fit_coa(X(b,:));
                p_best_f(b)=f; p_best_X(b,:)=X(b,:);
                if f>g_best_F
                        g_best_F=f;
                        g_best_X=X(b,:);
                 end
            end
        end
    end

    % 使用细网格进行二次复核，并用二分法提高精度
    S=deco(g_best_X);
    [drop_pts,blow_pts]=drop_blows_from_S(S,FY1,g,R_yun); 

    % 整体遮蔽的标志（ mask）
    covered=mask_any_cover(S,tGrid_fine,P_fine,M1,v_dao,uM,blow_pts,vd,R_yun,safety_ep);
    % 二分细化边界
    pred_any=@(t) cover_any_at_time(S,t,P_fine,M1,v_dao,uM,blow_pts,vd,R_yun,safety_ep);
    [interval_ref,total_cover_ref]=refine_mask_with_bisection( ...
        covered,tGrid_fine,pred_any,t_edge,max_b_it);

    % 判断三枚烟雾干扰各自生效的时长
    effDur=zeros(3,1);
    for gi=1:3
        covered_i=mask_one_cover(S,gi,tGrid_fine,P_fine,M1,v_dao,uM,blow_pts,vd,R_yun,safety_ep);
        pred_i=@(t) cover_one_at_time(S,gi,t,P_fine,M1,v_dao,uM,blow_pts,vd,R_yun,safety_ep);
        [~, dur_i]=refine_mask_with_bisection(covered_i,tGrid_fine,pred_i,t_edge,max_b_it);
        effDur(gi)=dur_i;
    end

    if total_cover_ref>best_all.f
        best_all.f=total_cover_ref;
        best_all.x=g_best_X;
        best_all.S=S;
        best_all.drop_pts=drop_pts;
        best_all.blow_pts=blow_pts;
        best_all.D_B_G=effDur;
        best_all.cover_Mask=covered;
        best_all.intervals=interval_ref;
        best_all.totalCover=total_cover_ref;
    end

    fprintf('[重启 %d/%d] 复核后=%.3f s\n',rs,c_f_cishu,total_cover_ref);
end

%进行结果的输出
Sbest=best_all.S;
drop_pts=best_all.drop_pts;
blow_pts=best_all.blow_pts;
effDur=best_all.D_B_G;
psi_deg=mod(Sbest.psi*180/pi, 360);
intervals=best_all.intervals;
total_cover=best_all.totalCover;

fprintf('\n===即将输出运行得到的最优解===\n');
fprintf('累计的完全遮蔽时长=%.3f s\n',total_cover);
fprintf('遮蔽区间并集：');
for k=1:size(intervals,1), fprintf('[%.3f, %.3f] ', intervals(k,1), intervals(k,2)); end
fprintf('\n航向(°)=%.3f | 速度=%.3f m/s\n', psi_deg, Sbest.vF);
for i=1:3
    fprintf('弹#%d: 投放 t=%.3f s, 延时=%.3f s, 起爆 t=%.3f s, 单弹有效=%.3f s\n', ...
        i, Sbest.d(i), Sbest.dim(i), Sbest.e(i), effDur(i));
end