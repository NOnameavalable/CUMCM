%问题四解决方案，使用PSO-DE加多次重启，依靠大量计算得出最终的解决
clear; clc;

%设置需要用的参数（可以根据情况调整）
% 在圆柱体的表面上采样
s_th = 48;
s_z = 10;  %在侧面上取的点，48*10
c_th = 48;
c_R  = 6; % 上下两个底面：48*6
safe_ep = 0.10;      % 安全裕度取0.1m
% 建立快速搜索和精搜的时间网格
dt_kuai = 0.30;     % 快速搜索阶段步长
dt_jing   = 0.02;         % 复核期步长（秒，越小越精）
tol_eg  = 1e-3; 
max_b_it = 40;

% PSO-DE 参数（可按机器加大）
dim= 12;   % 变量的维度：3*4共12个
n_p = 300;      % 种群规模
max_it  = 60;        % 最大迭代次数
el_rt  = 0.30;       % 设置精英的比例
Fm = 0.6;
CR = 0.7;  % 定义DE的缩放与交叉率
w_max = 0.9;
w_min = 0.4; % 采取线性惯性权重下降
c1 = 2;
c2 = 2;          %设置个体和社会的学习因子
im_e = 25;    % 设置每隔25代执行移民
im_rt = 0.20;             % 设置移民比例为20%
n_cishu  = 30;      % 设置重启次数为30次

%定义物理模型所需要的物理量
g      = 9.81;  %定义重力加速度
R_yun = 10;                  % 定义烟幕半径为10m
vd  = 3;         % 起爆后云心下沉速度 为3(m/s)

%设置导弹的参数
M1 = [20000, 0, 2000];  %M1的初始位置
v_dao  = 300;   %导弹速度为300m/s
uM = (-M1) / norm(M1);  %导弹M1速度方向的单位向量
t_end = norm(M1)/v_dao;    %求出最终时长

% 设置真目标的参数
r_xy = [0, 200, 0]; %下底面圆心的坐标
r_R = 7;  %半径
r_z = 10;  %高度


% 三架无人机FT1,FY2,FY的初始位置
rF0_all = [
    17800,     0, 1800;       % FY1
    12000,  1400, 1400;       % FY2
     6000, -3000,  700        % FY3
];

%建立时间网格
tGrid_coarse = 0:dt_kuai:t_end;
P_coarse = sampleCylinderPointsHP(r_xy, r_R, r_z, 24, 6, 24, 4); % 使用一个辅助函数，用于搜索期中等密度


tGrid_fine = 0:dt_jing:t_end;
P_fine = sampleCylinderPointsHP(r_xy, r_R, r_z, s_th, s_z, c_th, c_R); % 再次使用，用于复核期高密度

%使用自定义的辅助函数，完成粗评估
decode = @(x) decodeVars_Q4(x, t_end); % 12维[0,1] -> 3机(psi,v,d,Δ,e)
fitness_coarse = @(x) objective_cover_time_vec_Q4( ...
    x, decode, M1, v_dao, uM, rF0_all, R_yun, vd, g, ...
    tGrid_coarse, P_coarse, safe_ep);

%核心算法：PSO-DE+多次重启求出最优解
best_all = struct('f',-inf,'x',[],'S',[],'dropPts',[],'blowPts',[], ...
    'effDur',[],'intervals',[],'totalCover',[]);

rng('shuffle');
for rs = 1:n_cishu
    % 对粒子进行初始化
    X = rand(n_p, dim);
    V = 0.2*(rand(n_p,dim)-0.5);
    pbestX = X; pbestF = -inf(n_p,1);  %初始化个人最优
    gbestX = zeros(1,dim); gbestF = -inf;  %初始化群体最优


    %进行初评估
    for i=1:n_p
     %计算出适应度的同时检验是否符合约束条件
        f = fitness_coarse(X(i,:));
        
        % 通过飞行时间和烟雾弹下落时间来进行约束
        if X(i,7) < 0 || X(i,7) > 67 || X(i,8) < 0 || X(i,8) > 67 || X(i,9) < 0 || X(i,9) > 67
            f = -inf; % 不在约束内的解直接判定为最差解
        end
  
        
        if X(i,7) > 19.17 || X(i,8) > 16.9 || X(i,9) > 11.95
            f = -inf; % 不在约束内的解直接判定为最差解
        end
            %进行个体最优和群体最优的更新
        pbestF(i) = f; pbestX(i,:) = X(i,:);
        if f > gbestF, gbestF = f; gbestX = X(i,:); end
    end

    %进行迭代运行，迭代次数不超过最大迭代数
    for it = 1:max_it
        w = w_max - (w_max-w_min)*(it-1)/(max_it-1);

       %进行PSO过程
        for i=1:n_p
            r1 = rand(1,dim); r2 = rand(1,dim);
            V(i,:) = w*V(i,:) + c1*r1.*(pbestX(i,:) - X(i,:)) + c2*r2.*(gbestX - X(i,:));%更新粒子状态
            V(i,:) = max(min(V(i,:), 0.5), -0.5);
            X(i,:) = max(min(X(i,:) + V(i,:), 1), 0);
            
            % 通过飞行时间和烟雾弹下落时间来进行约束
            if X(i,7) < 0 || X(i,7) > 67 || X(i,8) < 0 || X(i,8) > 67 || X(i,9) < 0 || X(i,9) > 67
                f = -inf;
            end
            
            if X(i,7) > 19.17 || X(i,8) > 16.9 || X(i,9) > 11.95
                f = -inf;
            end
            
           % 计算出粒子适应度并进行更新
            f = fitness_coarse(X(i,:));
            if f > pbestF(i)
                pbestF(i) = f; pbestX(i,:) = X(i,:);
                if f > gbestF, gbestF = f; gbestX = X(i,:); end
            end
        end

        % 精英DE，交叉和缩放过程
        [~, idx] = sort(pbestF,'descend');
        nElite = max(3, round(el_rt*n_p));
        eliteIdx = idx(1:nElite);
        for e = eliteIdx'
            r = randperm(n_p,2); while any(r==e), r = randperm(n_p,2); end
            xr1 = X(r(1),:); xr2 = X(r(2),:); x = X(e,:);
            vDE = x + Fm*(gbestX - x) + Fm*(xr1 - xr2);
            jrand = randi(dim);
            u = x;
            for j=1:dim, if rand<=CR || j==jrand, u(j)=vDE(j); end, end
            u = max(min(u,1),0);
            fu = fitness_coarse(u);
            if fu > pbestF(e)
                X(e,:)=u; pbestX(e,:)=u; pbestF(e)=fu;
                if fu > gbestF, gbestF = fu; gbestX = u; end
            end
        end

        % 进行移民过程，将最差的20%移走
        if mod(it, im_e)==0
            [~, idxAsc] = sort(pbestF,'ascend');
            nImm = max(1, round(im_rt*n_p));
            badIdx = idxAsc(1:nImm);
            X(badIdx,:) = rand(nImm, dim);
            V(badIdx,:) = 0.2*(rand(nImm,dim)-0.5);
            for b = badIdx'
                f = fitness_coarse(X(b,:));
                pbestF(b) = f; pbestX(b,:) = X(b,:);
                if f > gbestF, gbestF = f; gbestX = X(b,:); end
            end
        end
    end

    % 使用自定以的辅助函数，生成精搜阶段的网格
    S = decode(gbestX);
    [dropPts, blowPts] = drop_blows_from_S_Q4(S, rF0_all, g);

    % 使用辅助函数，进行并集
    covered = mask_any_cover_Q4(S, tGrid_fine, P_fine, M1, v_dao, uM, blowPts, vd, R_yun, safe_ep);

    % 使用二分法细化
    pred_any = @(t) cover_any_at_time_Q4(S, t, P_fine, M1, v_dao, uM, blowPts, vd, R_yun, safe_ep);
    [intervals_ref, total_cover_ref] = refine_mask_with_bisection( ...
        covered, tGrid_fine, pred_any, tol_eg, max_b_it);

   % 使用二分法和独立判断确定每一枚烟雾弹单独的作用时间 
    effDur = zeros(3,1);
    for gi=1:3
        covered_i = mask_one_cover_Q4(S, gi, tGrid_fine, P_fine, M1, v_dao, uM, blowPts, vd, R_yun, safe_ep);
        pred_i = @(t) cover_one_at_time_Q4(S, gi, t, P_fine, M1, v_dao, uM, blowPts, vd, R_yun, safe_ep);
        [~, dur_i] = refine_mask_with_bisection(covered_i, tGrid_fine, pred_i, tol_eg, max_b_it);
        effDur(gi) = dur_i;
    end

   % 将全局最优解进行记录
    if total_cover_ref > best_all.f
        best_all.f=total_cover_ref;
        best_all.x=gbestX;
        best_all.S=S;
        best_all.dropPts=dropPts;
        best_all.blowPts=blowPts;
        best_all.effDur=effDur;
        best_all.intervals =intervals_ref;
        best_all.totalCover =total_cover_ref;
    end

    fprintf('[重启 %d/%d] 复核 = %.3f s\n', rs, n_cishu, total_cover_ref);
end

%将运行得到的最优解进行输出并保存表格
Sbest =best_all.S;
dropPts =best_all.dropPts;
blowPts =best_all.blowPts;
effDur  =best_all.effDur;
intervals =best_all.intervals;
total_cover=best_all.totalCover;

%输出摘要
fprintf('圆柱体累计完全遮蔽时长 = %.3f s\n', total_cover);
fprintf('有效的遮蔽时间区间并集：');
for k=1:size(intervals,1), fprintf('[%.3f, %.3f] ', intervals(k,1), intervals(k,2)); end
fprintf('\n');
for i=1:3
    psi_deg = mod(Sbest.psi(i)*180/pi, 360);
    fprintf('FY%d: 航向=%.2f° | 速度=%.2f m/s | 投放 d=%.2f s, 延时 Δ=%.2f s, 起爆 e=%.2f s | 单弹完全遮蔽=%.2f s\n', ...
        i, psi_deg, Sbest.vF(i), Sbest.d(i), Sbest.dim(i), Sbest.e(i), effDur(i));
end


% 将需要的数据写入excel表格中
hdr = { ...
    '无人机编号', ...
    '无人机运动方向 (°)', ...
    '无人机运动速度 (m/s)', ...
    '烟幕干扰弹投放点的x坐标 (m)', ...
    '烟幕干扰弹投放点的y坐标 (m)', ...
    '烟幕干扰弹投放点的z坐标 (m)', ...
    '烟幕干扰弹起爆点的x坐标 (m)', ...
    '烟幕干扰弹起爆点的y坐标 (m)', ...
    '烟幕干扰弹起爆点的z坐标 (m)', ...
    '有效干扰时长 (s)'}; 

rows = cell(3, numel(hdr));
for i=1:3
    rows{i,1} = sprintf('FY%d', i);
    rows{i,2} = mod(Sbest.psi(i)*180/pi, 360);
    rows{i,3} = Sbest.vF(i);
    rows{i,4} = dropPts(i,1);
    rows{i,5} = dropPts(i,2);
    rows{i,6} = dropPts(i,3);
    rows{i,7} = blowPts(i,1);
    rows{i,8} = blowPts(i,2);
    rows{i,9} = blowPts(i,3);
    rows{i,10}= effDur(i);
end
outfile = 'result2.xlsx';
if exist(outfile,'file'), delete(outfile); end
writecell(hdr, outfile, 'Sheet', 1, 'Range', 'A1');
writecell(rows, outfile, 'Sheet', 1, 'Range', 'A2');
fprintf('已写出 Excel: %s\n', outfile); %说明已经写入到excel表格中


%使用的自定义的函数
function S=decodeVars_Q4(x, t_end)
    x = x(:).';
    if numel(x) ~= 12, error('维度不符'); end     %判断是否为12维
    psi = zeros(1,3); vF = zeros(1,3); d = zeros(1,3); dim = zeros(1,3); e = zeros(1,3);
    for k = 1:3
        u_psi= x(4*(k-1)+1);
        u_v = x(4*(k-1)+2);
        u_d= x(4*(k-1)+3);
        u_D= x(4*(k-1)+4);
        psi(k)= 2*pi*u_psi;
        vF(k)= 70 + 70*u_v;          
        d(k)= 0 + t_end*u_d;       
        dim(k)= 20*u_D;               
        e(k)= d(k) + dim(k);
    end
    S = struct('psi',psi, 'vF',vF, 'd',d, 'dim',dim, 'e',e);
end

function [coverTime] = objective_cover_time_vec_Q4(x, decode, M1, v_dao, uM, rF0_all, ...
        R_yun, vd, g, tGrid, P, safe_ep)
        % 粗评估目标函数：返回并集 完全遮蔽 累计时长（秒）
    S = decode(x);
    [~, blow] = drop_blows_from_S_Q4(S, rF0_all, g);
    covered = mask_any_cover_Q4(S, tGrid, P, M1, v_dao, uM, blow, vd, R_yun, safe_ep);
    coverTime = sum(covered) * (tGrid(2)-tGrid(1));
end

function [dropPts, blowPts] = drop_blows_from_S_Q4(S, rF0_all, g)
        % 该函数用于计算出FY1，FY2，FY3的投放点和起爆点
    dropPts = zeros(3,3); blowPts = zeros(3,3);
    for i=1:3
        uF = [cos(S.psi(i)), sin(S.psi(i)), 0];
        p0 = rF0_all(i,:);
        dropPts(i,:) = p0 + S.vF(i)*S.d(i)*uF; dropPts(i,3) = p0(3);
         % 起爆点：延时期间视为从投掷点开始的平抛运动
        tau = S.dim(i);
        blowPts(i,1:2) = dropPts(i,1:2) + S.vF(i)*tau*uF(1:2);
        blowPts(i,3)   = dropPts(i,3) - 0.5*g*tau^2;
    end
end

function covered = mask_any_cover_Q4(S, tGrid, P, M1, v_dao, uM, blow, vd, R_yun, safe_ep)
    % 这函数是用于求三个遮蔽时间集合的并集
    covered = false(size(tGrid));
    for k=1:numel(tGrid)   %使用循坏实现函数功能
        t = tGrid(k);
        rM = M1 + v_dao*t*uM;
        anyOn = false;
        for i=1:3
            if t>=S.e(i) && t<=S.e(i)+20
                c = blow(i,:) + [0,0,-vd*(t - S.e(i))];
                if cloudFullyCovers_vec(c, R_yun, rM, P, safe_ep)
                    anyOn = true; break;
                end
            end
        end
        covered(k) = anyOn;
    end
end

function covered = mask_one_cover_Q4(S, gi, tGrid, P, M1, v_dao, uM, blow, vd, R_yun, safe_ep)
    % 求出一个烟雾干扰弹的有效干扰时间区间
    covered = false(size(tGrid));
    for k=1:numel(tGrid)
        t = tGrid(k);
        if t>=S.e(gi) && t<=S.e(gi)+20
            rM = M1 + v_dao*t*uM;
            c  = blow(gi,:) + [0,0,-vd*(t - S.e(gi))];
            covered(k) = cloudFullyCovers_vec(c, R_yun, rM, P, safe_ep);
        end
    end
end

function tf = cover_any_at_time_Q4(S, t, P, M1, v_dao, uM, blow, vd, R_yun, safe_ep)
   % 使用二分化细化，求出遮蔽区间的首尾时刻
    rM = M1 + v_dao*t*uM;
    tf = false;
    for i=1:3
        if t>=S.e(i) && t<=S.e(i)+20
            c = blow(i,:) + [0,0,-vd*(t - S.e(i))];
            if cloudFullyCovers_vec(c, R_yun, rM, P, safe_ep)
                tf = true; return;
            end
        end
    end
end

function tf = cover_one_at_time_Q4(S, gi, t, P, M1, v_dao, uM, blow, vd, R_yun, safe_ep)
   % 该函数是表示用二分法求出每一给烟雾弹的遮蔽首位时刻
    tf = false;
    if t>=S.e(gi) && t<=S.e(gi)+20
        rM = M1 + v_dao*t*uM;
        c  = blow(gi,:) + [0,0,-vd*(t - S.e(gi))];
        tf = cloudFullyCovers_vec(c, R_yun, rM, P, safe_ep);
    end
end

function tf = cloudFullyCovers_vec(c, R, rM, P, safe_ep)
     %该函数用于判断目标是否被烟雾完全遮蔽
    R_eff = R - safe_ep;                 % 收紧半径，增强数值的稳定性
    AB = P - rM;                            
    AB2 = sum(AB.*AB, 2);                  
    CA = c - rM;                            
    t  = zeros(size(AB2));
    nz = AB2 > 1e-12;
    t(nz) = (AB(nz,:)*CA.')./AB2(nz);       % 计算出投影的参数
    t = max(0, min(1, t));                  % 找出clamp 到线段
    Q = rM + t.*AB;                         % 找到最近点
    d = sqrt(sum((c - Q).^2, 2));           % 求出点到直线的距离
    tf = all(d <= R_eff + 1e-12);
end

function [intervals, total_len] = refine_mask_with_bisection(mask, tGrid, pred, tol, maxit)
   % 该函数作用是得到更精确的遮蔽时间区间
    dt = tGrid(2) - tGrid(1);
    idx = find(mask);
    intervals = zeros(0,2);
    total_len = 0;
    if isempty(idx), return; end
    % 按照真的段分组
    jumps = [1, find(diff(idx)>1)+1, numel(idx)+1];
    for s = 1:numel(jumps)-1
        seg = idx(jumps(s):jumps(s+1)-1);
        tR = tGrid(seg(1));
        tL = max(0, tR - dt);
        bR = pred(tR); bL = pred(tL);
        if ~bR && bL, tmp=tL; tL=tR; tR=tmp; tmp=bL; bL=bR; bR=tmp; end
        if bL && bR
            t_left = tL;
        else
            for k=1:maxit
                tm = 0.5*(tL+tR);
                if pred(tm)==bR, tR=tm; else, tL=tm; end
                if (tR - tL) <= tol, break; end
            end
            t_left = tR;
        end

        tL = tGrid(seg(end));
        tR = min(tGrid(end), tL + dt);
        bL = pred(tL); bR = pred(tR);
        if bL && bR
            t_right = tR;
        else
            for k=1:maxit
                tm = 0.5*(tL+tR);
                if pred(tm)==bL, tL=tm; else, tR=tm; end
                if (tR - tL) <= tol, break; end
            end
            t_right = tL;
        end

        intervals(end+1,:) = [t_left, t_right];
        total_len = total_len + (t_right - t_left);
    end
end

function P = sampleCylinderPointsHP(c0, R, H, nTh_side, nZ_side, nTh_cap, nRad_cap)
    % 函数用于圆柱表面采样，在圆柱体表面采取若干个点
    
    % 下面是采取侧面点

    the = linspace(0,2*pi,nTh_side+1); the(end)=[];
    zs  = linspace(c0(3), c0(3)+H, nZ_side);
    P_side = zeros(nTh_side*nZ_side,3); idx=0;
    for z = zs
        for th = the
            idx=idx+1; P_side(idx,:) = [c0(1)+R*cos(th), c0(2)+R*sin(th), z];
        end
    end
    % 上/下底
    rs  = linspace(0,R,nRad_cap);
    the2= linspace(0,2*pi,nTh_cap+1); the2(end)=[];
    P_cap = zeros(numel(rs)*numel(the2)*2,3); idx2=0;
    for z = [c0(3), c0(3)+H]
        for r = rs
            for th = the2
                idx2=idx2+1; P_cap(idx2,:) = [c0(1)+r*cos(th), c0(2)+r*sin(th), z];
            end
        end
    end
    % 下面表示的是中高度的额外极值
    extra = [c0(1)+R, c0(2),     c0(3)+H/2;
             c0(1)-R, c0(2),     c0(3)+H/2;
             c0(1),   c0(2)+R,   c0(3)+H/2;
             c0(1),   c0(2)-R,   c0(3)+H/2];
    P = unique([P_side; P_cap(1:idx2,:); extra], 'rows');
end