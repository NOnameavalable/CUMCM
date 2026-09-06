clear; clc;
%设置需要使用的常量，单位是m,s
FY1=[17800, 0, 1800]; % 无人机FY1的初始位置
M1=[20000, 0, 2000];  % 导弹M1的初始位置
v_F=120;              % 规定无人机速度
v_dao=300;              % 规定导弹速度
td=1.5;               % 规定投烟雾弹的放时刻
tau=3.6;              % 规定烟雾弹空中降落时间
te=td + tau;          % 起爆时刻
vd=3;              % 设置云团匀速下落的速度
R_yun=10;             % 设置云团的有效半径
g=9.8;                % 设置重力加速度
r_R=7;                 %规定圆柱体的半径
r_z=[0, 10];         %圆柱体的高
r_xy=[0, 200];        %圆柱体的圆心

%计算FY1朝向原点的水平单位向量
d_h=-[FY1(1), FY1(2), 0];
d_h=d_h/norm(d_h);
S=FY1+v_F*td*d_h;            % 投放点S
E=S+v_F*tau*d_h+0.5*[0,0,-g]*tau^2;      %起爆点E

%在圆柱体上采样，通过采样多个点近似于整个圆柱体
n_r=30;        %在圆柱体径向的采样数
n_z=30;         %在圆柱高度向的采样数
nth=16;       %在圆柱体周向的采样数
include_b=false;        %是否包含边界点

[interval,T_total,info]=cover_time_cylinder_sampling_( ...
    M1, v_dao, E, vd, te, R_yun, r_xy, r_R, r_z, ...
    n_r, nth, n_z, include_b);

% 输出计算出的各项结果
format long g
fprintf('采样点总数:%d (n_r=%d,Nθ=%d,n_z=%d,边界=%d)\n', ...
    info.n_p, n_r, nth, n_z, include_b);

if isempty(interval)         %判断是否为0
    disp('圆柱体在任何时刻都未被完全遮蔽。');
else
    fprintf('圆柱体被完全遮蔽的时间区间（共%d段)：\n', size(interval,1));
    for k=1:size(interval,1)
        fprintf('  第%02d段: [%.9f , %.9f] s(时长=%.9f s)\n', ...
            k, interval(k,1), interval(k,2), interval(k,2)-interval(k,1));
    end
    fprintf('\n圆柱体的有效遮蔽总时长:T_all=%.9f s\n', T_total);
end
evt=missile_cloud_events_(M1, v_dao, E, te, vd, R_yun);  

fprintf('\n导弹M1与云团的相对位置关系：\n');
fprintf('云团有效的时间区间：[%.6f，%.6f] s\n', evt.t_window(1), evt.t_window(2));
if ~evt.has_intersection        %判断导弹在云团里的时间是否为0
    fprintf('导弹M1未进入云团。\n');
else
    fprintf('导弹M1进入云团时刻：t_in=%.9f s\n', evt.t_in);
    fprintf('导弹M1脱离云团时刻：t_out=%.9f s\n', evt.t_out);
    fprintf('M1在云内经历时间：Δt=%.9f s\n', evt.duration);
end
