%问题2：FY1无人机投放1枚烟雾干扰弹最优方案
clear;clc;

%定义需要使用的量
g=9.8;   %定义重力加速度
M1=[20000,0,2000];   %定义M1的初始位置
v_dao=300;   %定义导弹的速度
R_yun=10;   %定义云团的有效半径
v_yun=3;    %定义云团匀速下落的速度
r_R=7;          %规定圆柱体的半径
r_xy=[0,200];        %圆柱体的圆心
r_z=[0,10];      %圆柱体的高

%在圆柱体上采样，通过采样多个点近似于整个圆柱体
n_r=4;        %在圆柱体径向的采样数
n_z=4;         %在圆柱高度向的采样数
nth=16;       %在圆柱体周向的采样数
include_b=false;        %是否包含侧壁/端面边界点

% 设置决策变量边界[速度, 角度, 飞行时间, 空中下降时间]
lb=[ 70,  0,  0,  0];            %设置下界
ub=[140, 2*pi, 60, 15];        %设置上界

% 设置PSO参数
n_p=60;              % 设置粒子数
w0=0.9; w1 = 0.40;        % 设置惯性权重，采取线性递减
c1=1.49; c2 = 2;       % 设置个体与社会的学习因子
i_t=60;             % 设置算法的迭代次数

% 初始化过程
dim=4;     %四个维度
rng('shuffle');
v=zeros(n_p,dim);                % 初始化粒子的速度
x=lb+(ub-lb).*rand(n_p,dim);     % 初始化粒子的位置

%评估初值,使用循环，实现PSO最小化
fit=zeros(n_p,1);
for i=1:n_p
    fit(i)=-q2_objective_cylinder_( x(i,:),M1,v_dao,g,v_yun,R_yun, ...
        r_xy,r_R,r_z,n_r,nth,n_z,include_b );
end
Pbest=x;   PbestVal=fit;     %个体最优
[GbVal, gid]=min(fit);         % 最小化步骤
Gbest=x(gid,:);                   %群体最优

% 主循环
for it=1:i_t
    w=w0 + (w1 - w0) * (it-1)/(i_t-1);
    for i=1:n_p
        r1=rand(1,dim); r2 = rand(1,dim);
        v(i,:)=w*v(i,:) + c1*r1.*(Pbest(i,:) - x(i,:)) + c2*r2.*(Gbest - x(i,:));
        x(i,:)=x(i,:) + v(i,:);
        % 处理位置边界
        x(i,:)=max(lb, min(ub, x(i,:)));
        % 进行评估
        fi=-q2_objective_cylinder_( x(i,:), M1, v_dao, g, v_yun, R_yun, ...
            r_xy, r_R, r_z, n_r, nth, n_z, include_b );
        fit(i)=fi;
        % 更新个体和全局最优
        if fi < PbestVal(i), PbestVal(i) = fi; Pbest(i,:) = x(i,:); end
        if fi < GbVal, GbVal = fi; Gbest = x(i,:); end
    end
    fprintf('i_ter %3d: -bestObj = %.6f s\n', it, -GbVal);
end

% —— 输出最优解与验证（用圆柱采样精确器） —— 
v=Gbest(1);   phi=Gbest(2);   td=Gbest(3);   tau=Gbest(4);
te=td + tau;
d_h=[cos(phi), sin(phi), 0];                   % FY1的飞行方向
FY1=[17800,0,1800];                              %FY1的初始位置
S=FY1 + v*td*d_h;                               % 投放点
E=S + v*tau*d_h + 0.5*[0,0,-g]*tau^2;          % 起爆点

[interval, T_total, info]=cover_time_cylinder_sampling( ...
    M1,v_dao,E,v_yun,te,R_yun,r_xy,r_R,r_z, ...
    n_r,nth,n_z,include_b );

% 输出结果
fprintf('v=%.3f m/s,phi=%.3f rad(%.2f°)\n', v,phi,phi*180/pi);
fprintf('t_d=%.3f s,tau=%.3f s,t_e=%.3f s\n',td,tau,te);
fprintf('E=(%.3f,%.3f,%.3f)\n', E);
fprintf('采样点数目: %d  (n_r=%d,Nθ=%d,n_z=%d,边界=%d)\n', ...
    info.n_points,n_r,nth,n_z,include_b);
if isempty(interval)
    fprintf('没有任何时刻能将真目标遮蔽完全。\n');
else
    fprintf('真目标被完全遮蔽的有效时间区间（%.9f）:\n', size(interval,1));
    for k = 1:size(interval,1)
        fprintf('  区间%02d: [%.9f , %.9f] (%.9f s)\n', ...
            k, interval(k,1), interval(k,2), interval(k,2)-interval(k,1));
    end
end
fprintf('\n有效遮蔽真目标的总时长: T_cov = %.9f s\n', T_total);

function evt=missile_cloud_events_(M1,v_dao,E,te,vd,R_yun)
    uM=-M1 / norm(M1);   %按照问题一的描述，计算导弹M1指向原点方向的单位向量
    k=[0,0,1];     % 给出二次式系数，因为云团下落，所以x,y取为0
    E_s=E+vd*te*k;   %起爆后的云团位置
    v_c=v_dao*uM+vd*k;          % 计算相对速度
    R0=M1-E_s;              % 初始相对位移
    %计算连线到云团中心的距离
    a=dot(v_c,v_c);  
    b=2*dot(R0,v_c);
    c=dot(R0,R0);             
    c=c-R_yun^2;% 将式子写成一元二次方程的形式，如f(x)=ax^2+bx+(c-R^2)
    evt=struct('has_intersection',false, 't_window',[te, te+20], ...
                 't_in',[], 't_out',[], 'duration',0, ...
                 't_raw',[NaN,NaN], 'a',a, 'b',b, 'c',c, 'disc',NaN);
    if a<1e-14   %判断相对速度是否为0
        if c<=0      %判断相对距离是否小于等于0，即导弹是否在云内
            evt.has_intersection=true;  %逻辑值1，表示在云内
            evt.t_raw=[-inf, +inf];
            evt.t_in=te;
            evt.t_out=te+20;
            evt.duration=20;
        end
        return;
    end
    %相对速度不为0的情况
    D=b*b - 4*a*c;     %计算一元二次方程的判别式子
    evt.disc = D;
    if D < 0      %方程无解，即没有交集
        return;
    end

    %方程有解,假设t1<=t2
    sq=sqrt(max(D,0));
    t1=(-b-sq)/(2*a);  %即开始遮蔽时刻
    t2=(-b+sq)/(2*a);   %即结束遮蔽时刻
    if t1>t2, tmp=t1; t1=t2; t2=tmp; end  %使假设t1<=t2成立
    evt.t_raw=[t1, t2]; %遮蔽时间段
    % 云团有效时间取交
    A=te;
    B=te+20;
    tin=max(t1, A);
    tout=min(t2, B);

    if tin<=tout  %不成立为空
        evt.has_intersection=true;
        evt.t_in=tin;
        evt.t_out=tout;
        evt.duration=max(0, tout - tin);
    end
end
function [interval,T_total]=cover_time_point_target_(T,M1,v_dao,E,vd,te,R_yun)
    uM=-M1/norm(M1);%按照问题一的描述，计算导弹M1指向原点方向的单位向量
    %划分时间网格
    t0=te; t1=te+20;
    dt=1e-3;  %时间步长
    t=(t0:dt:t1).';
    %按时间顺序排序
    if t(end)<t1, t=[t;t1]; end

    %计算导弹M1每个时刻的坐标
    M1_x=M1(1)+v_dao*uM(1).*t;
    M1_y=M1(2)+v_dao*uM(2).*t;
    M1_z=M1(3)+v_dao*uM(3).*t;
    %计算云团每个时刻的坐标
    C_x=E(1)+0.*t;
    C_y=E(2)+0.*t;
    C_z=E(3)-vd.*(t-te);

    %用向量的方法，计算点到直线的距离
    A=T(:).';
    AB=[M1_x-A(1),M1_y-A(2),M1_z-A(3)];   %取线段AB
    CA=[C_x-A(1),C_y-A(2),C_z-A(3)];      %取线段CA
    L=sqrt(sum(AB.^2, 2));   %向量AB的模
    dotC=sum(CA.*AB, 2);
    Lambda=dotC./(L.^2);      %夹角的三角函数值                     
    cr=cross(CA, AB,2);
    ju_li=sqrt(sum(cr.^2,2)) ./ L;  %计算距离

    dA=sqrt(sum(CA.^2,2));  %向量CA的模
    CB=[C_x - M1_x, C_y-M1_y, C_z-M1_z];  %取一线段CB
    dB=sqrt(sum(CB.^2,2));  %向量CB的模

    panduan=(Lambda>= 0) & (Lambda <= 1);%投影是否在线段内
    d=ju_li;
    d(~panduan)=min(dA(~panduan), dB(~panduan));
    %检验标志
    mask=(d<=R_yun+1e-12);
    % 找边界，再进行细化
    edge=diff([0; mask; 0]);       % +1表示进入；-1 离开
    iL=find(edge == +1);
    iR=find(edge == -1) - 1;

    interval=zeros(numel(iL), 2);
    for k=1:numel(iL)
        ia=iL(k); ib=iR(k);
        tL=bisect_(@(x) d_at_(x)-R_yun, t(max(ia-1,1)), t(ia), 1e-10, 80);
        tR=bisect_(@(x) d_at_(x)-R_yun, t(ib), t(min(ib+1,numel(t))), 1e-10, 80);
        interval(k,:)=[tL, tR];
    end

    % 求出总时长
    T_total=sum(interval(:,2) - interval(:,1), 'omitnan');

    function val=d_at_(x)
        %计算导弹M1每个时刻的坐标
        Mx=M1(1)+v_dao*uM(1)*x;
        My=M1(2)+v_dao*uM(2)*x;
        Mz=M1(3)+v_dao*uM(3)*x;
        %计算云团每个时刻的坐标
        Cx=E(1);
        Cy=E(2);
        Cz=E(3)-vd*(x-te);

        AB1=[Mx-A(1),  My-A(2),  Mz-A(3)];    %取线段AB1
        CA1=[Cx-A(1),  Cy-A(2),  Cz-A(3)];     %取线段CA1
        L1 =norm(AB1);
        s1 =dot(CA1, AB1) / L1;              
        d_line1=norm(cross(CA1, AB1)) / L1;  %计算出直线距离

        if s1 >= 0 && s1 <= L1
            val=d_line1;                     % 线段内取点
        else
            val=min( norm(CA1), norm([Cx-Mx, Cy-My, Cz-Mz]) ); % 线段外取端点
        end
    end
end

%使用简单的二分法
function r=bisect_(fun, a, b, tol, itmax)
    fa=fun(a); fb=fun(b);
    if sign(fa)==0, r=a; return; end
    if sign(fb)==0, r=b; return; end
    if fa*fb > 0
        % 假如同号，那么就返回较小值所在的端点
        r=(abs(fa) <= abs(fb))*a +(abs(fa) > abs(fb))*b; 
        return;
    end
    for it=1:itmax
        m=0.5*(a+b); fm=fun(m);
        if fm==0 || (b-a) < tol, r=m; return; end
        if sign(fa)*sign(fm) <=0, b=m; fb=fm; else, a = m; fa = fm; end
    end
    r = 0.5*(a+b);
end


function [interval, T_total, info] = cover_time_cylinder_sampling_( ...
    M1,v_dao,E,vd,te,R_yun,r_xy,r_R,r_z, ...
    n_r,nth,n_z,include_b)
    %在圆柱体内生成若干采样点
    c_x=r_xy(1);
    c_y=r_xy(2);
    z1=r_z(1);
    z2=r_z(2);

    %在半径方向上均匀采点，主函数中说明include_b为false，即不取边界
    if include_b
        z_l=linspace(z1, z2, n_z);
        r_l=linspace(0, r_R, n_r);
    else
        u=linspace(1/(n_r+1), n_r/(n_r+1),n_r);
        r_l=r_R*sqrt(u);
        if n_z==1
            z_l=(z1+z2)/2;
        else
            z_l=linspace(z1,z2,n_z+2); 
            z_l=z_l(2:end-1);           % 去除端点
        end
    end
    the_l=linspace(0,2*pi,nth+1);
    the_l(end)=[]; %去除里面重复的值
    T_p=[];%采样点的集合，将去重后的采样点放入
    for iz=1:numel(z_l)
        z=z_l(iz);
        for ir=1:numel(r_l)
            r=r_l(ir);
            if r==0
                T_p(end+1,:)=[c_x,c_y,z];
            else
                for it=1:numel(the_l)
                    th=the_l(it);
                    x=c_x + r*cos(th);
                    y=c_y + r*sin(th);
                    T_p(end+1,:)=[x,y,z];
                end
            end
        end
    end

    % —— 2) 逐点求遮蔽区间并不断取“交集” —— 
    % 初始交集为整个有效窗
    interval=[te, te+20];
    for p=1:size(T_p,1)
        T=T_p(p,:);
        [interval_p,~]=cover_time_point_target(T,M1,v_dao, E,vd,te,R_yun);
        interval=intersect_interval_sets_(interval,interval_p);
        if isempty(interval)   %如果是空集了，直接跳出循坏
            break;  
        end
    end
    %输出总的时间
    if isempty(interval) 
        T_total=0;
    else
        T_total=sum(interval(:,2) - interval(:,1));
    end

    %将提取采样的信息存入info中
    info.the_l=the_l;
    info.n_p=size(T_p,1);
    info.z_l=z_l;
    info.include_b=include_b;
    info.r_l=r_l;
end

