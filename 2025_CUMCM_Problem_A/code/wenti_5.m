%第五问
clear; clc; close all; rng(1);

    %先进行参数的调试
DEBUG        = true;  % 打印出活跃窗上限、采样点数、单导弹时长等调试信息
USE_PARFOR   = true;  % 如果有并行工具箱就使用并行；没有的化会自动退化成串行
INJECT_SEEDS = true;   % 为了加块收敛，注入几条启发式种子

%设置问题中需要的常量
Rcloud=10;    %烟雾的半径10m
vsink=3;   %烟雾的下降速度3m/s
eff_win=20;
safety_eps=0.10;
cyl_center=[0,200,0];
cyl_R=7;
cyl_H=10;

% 三枚导弹的初始位置
M0 = [ 20000,    0, 2000;
        19000,  600, 2100;
        18000, -600, 1900 ];
vM = 300; %三枚导弹的初始飞行速度
% 导弹的飞行方向方向指向原点
uMO = bsxfun(@rdivide, -M0, vecnorm(M0,2,2));%导弹飞行方向的行向量
Tend = vecnorm(M0,2,2)/vM;

% 五个无人机的初始位置
UAV0 = [17800,     0, 1800;
        12000,  1400, 1400;
         6000, -3000,  700;
        11000,  2000, 1800;
        13000, -2000, 1300];

% 干扰导弹的分配规则：FY1→M1，FY4→M2，FY2/3/5→M3
assign_map = [1,3,3,2,3];

% 无人机的速度范围，70-140m/s内
vmin = 70; vmax = 140;

%对圆柱进行采样
P_sentinel = sampleCylinderPoints(cyl_center, cyl_R, cyl_H, 16, 3, 16, 2);   % ~112
P_medium   = sampleCylinderPoints(cyl_center, cyl_R, cyl_H, 24, 6, 24, 3);   % ~288
P_fine     = sampleCylinderPoints(cyl_center, cyl_R, cyl_H, 48,10, 48, 6);   % ~1056
R_eff = Rcloud - safety_eps;

dt_coarse = 0.25;      % 设置的值低一些，防止会存在因为整数导致的错觉
dt_fine   = 0.05;      % 最终的掩码与区间分辨率

%共有40维
% 每机 8 维：psi, v, (d,Δ)x3 所以5×8=40
D = 40; LB=zeros(1,D); UB=ones(1,D); Tmax = max(Tend);
for i=1:5
    b=(i-1)*8;
    LB(b+1)=0;     UB(b+1)=2*pi;     % 方位角
    LB(b+2)=vmin;  UB(b+2)=vmax;     % 速度
    for k=1:3
        LB(b+2+2*k-1)=0;   UB(b+2+2*k-1)=Tmax;  % 时间
        LB(b+2+2*k  )=0;   UB(b+2+2*k  )=20;    
    end
end

%使用PSO算法，设置PSO的参数
NP=300; maxIter=240; w_max=0.9; w_min=0.4; c1=2.2; c2=2.2; v_clamp=0.3;
n_restarts=20;  %包含粒子数，最大迭代次数，惯性权重等

%为程序实现并行的配置
if USE_PARFOR
    try
        if ~license('test','Distrib_Computing_Toolbox'), USE_PARFOR=false; end
        if USE_PARFOR && isempty(gcp('nocreate')), parpool('threads'); end
    catch
        USE_PARFOR=false;
    end
end

%程序的主循环部分
best = struct('J',-inf,'x',[],'detail',[]);
for rs=1:n_restarts
    X = rand(NP,D); V = 0.1*(rand(NP,D)-0.5);

  %采用启发式种子
    if INJECT_SEEDS
        nseed = min(6, NP);
        seeds = heuristic_seeds(nseed, LB, UB, UAV0);
        X(1:nseed,:) = seeds;
    end

    % 进行初次评价
    F = eval_population(X, LB, UB, assign_map, UAV0, M0, uMO, vM, ...
                        P_sentinel, P_medium, P_fine, R_eff, vsink, eff_win, Tend, dt_coarse, USE_PARFOR);
    pX = X; pF = F; [gF,gi]=max(F); gX = X(gi,:);

    for it=1:maxIter
        w = w_max - (w_max-w_min)*(it-1)/(maxIter-1);

        % 同时进行更新过程
        R1 = rand(NP,D); R2 = rand(NP,D);
        V = w.*V + c1.*R1.*(pX - X) + c2.*R2.*(gX - X);
        V = max(min(V, v_clamp), -v_clamp);
        X = min(max(X + V, 0), 1);

        % 再进行串行与并行的评估
        F = eval_population(X, LB, UB, assign_map, UAV0, M0, uMO, vM, ...
                            P_sentinel, P_medium, P_fine, R_eff, vsink, eff_win, Tend, dt_coarse, USE_PARFOR);

        % 再次进行更新
        better = F > pF; pF(better) = F(better); pX(better,:) = X(better,:);
        [curF, idx] = max(F);
        if curF > gF, gF = curF; gX = X(idx,:); end

        %进行轻量的重置
        if mod(it,25)==0
            [~,ord]=sort(F,'ascend'); nb=max(1,round(0.15*NP));
            bad=ord(1:nb); X(bad,:)=rand(nb,D); V(bad,:)=0;
            Fb = eval_population(X(bad,:), LB, UB, assign_map, UAV0, M0, uMO, vM, ...
                                 P_sentinel, P_medium, P_fine, R_eff, vsink, eff_win, Tend, dt_coarse, USE_PARFOR);
            pX(bad,:)=X(bad,:); pF(bad)=Fb;
            if max(Fb) > gF, [~,k]=max(Fb); gF=Fb(k); gX=X(bad(k),:); end
        end
    end

    % 对目前的全局最优解进行精搜
    detail = finalize_solution_nobis(gX, LB, UB, assign_map, UAV0, M0, uMO, vM, ...
        P_sentinel, P_medium, P_fine, R_eff, vsink, eff_win, Tend, dt_fine, DEBUG);

    if detail.Jsum > best.J, best.J=detail.Jsum; best.x=gX; best.detail=detail; end
    fprintf('Restart %d/%d : best(sum M1-3)=%.3f s\n', rs, n_restarts, best.J);
end

%将运行的结果输出到excel表格中
out = 'result3_fast_compat_nobis.xlsx';
if exist(out,'file'), delete(out); end

S = best.detail.S;
hdr = {'无人机','分配导弹','航向(°)','速度(m/s)','投放d(s)','延时Δ(s)','起爆e(s)', ...
       '投放x','投放y','投放z','起爆x','起爆y','起爆z'};
rows = {};
for i=1:5
    for b=1:3
        rows(end+1,:) = {sprintf('FY%d',i), sprintf('M%d',S.assign(i)), ...
            mod(rad2deg(S.psi(i)),360), S.vF(i), S.d(i,b), S.D(i,b), S.e(i,b), ...
            S.P_drop(i,b,1), S.P_drop(i,b,2), S.P_drop(i,b,3), ...
            S.P_blow(i,b,1), S.P_blow(i,b,2), S.P_blow(i,b,3)};
    end
end
writecell(hdr, out, 'Sheet','方案-全部', 'Range','A1');
writecell(rows, out, 'Sheet','方案-全部', 'Range','A2');

for m=1:3
    iv = best.detail.intervals{m};
    if isempty(iv), iv=zeros(0,2); end
    T_iv = array2table(iv, 'VariableNames', {'开始(s)','结束(s)'});
    writetable(T_iv, out, 'Sheet', sprintf('M%d-区间',m));
end
Tstat = table((1:3).', best.detail.Tsum(:), 'VariableNames', {'导弹','累计完全遮蔽时长(s)'});
Tstat2 = table(sum(best.detail.Tsum), 'VariableNames', {'三导弹累计和(s)'});
writetable(Tstat, out, 'Sheet','统计', 'Range','A1');
writetable(Tstat2, out, 'Sheet','统计', 'Range','D1');
fprintf('已写出 %s\n', out);

%所用到的自定义函数
function F = eval_population(X, LB, UB, assign_map, UAV0, M0, uMO, vM, ...
                             Psent, Pmed, Pfine, R_eff, vsink, eff_win, Tend, dt, USE_PARFOR)
% 有parfor/for 两种模式，可以进行批量处理
    NP = size(X,1); F = zeros(NP,1);
    if USE_PARFOR
        parfor i = 1:NP
            F(i) = objective_fast(X(i,:), LB, UB, assign_map, UAV0, M0, uMO, vM, ...
                                  Psent, Pmed, Pfine, R_eff, vsink, eff_win, Tend, dt);
        end
    else
        for i = 1:NP
            F(i) = objective_fast(X(i,:), LB, UB, assign_map, UAV0, M0, uMO, vM, ...
                                  Psent, Pmed, Pfine, R_eff, vsink, eff_win, Tend, dt);
        end
    end
end

function J = objective_fast(x, LB, UB, assign_map, UAV0, M0, uMO, vM, ...
                            Psent, Pmed, Pfine, R_eff, vsink, eff_win, Tend, dt)
%评估的目标函数，三个导弹在各自的活跃窗并采样
    S = decode_Q5(x, LB, UB, assign_map, UAV0);
    S = points_from_S(S, UAV0);
    J = 0;
    for m=1:3
        iv = active_windows(m, S, Tend(m), eff_win);
        if isempty(iv), continue; end
        covered_sum = 0;
        for k=1:size(iv,1)
            ts = iv(k,1):dt:min(iv(k,2),Tend(m));
            if isempty(ts), continue; end
            rM = bsxfun(@plus, M0(m,:), bsxfun(@times, vM*ts(:), uMO(m,:))); % (#t,3)
            for ti=1:numel(ts)
                if is_time_covered(ts(ti), rM(ti,:), m, S, Psent, Pmed, Pfine, R_eff, vsink, eff_win)
                    covered_sum = covered_sum + 1;
                end
            end
        end
        J = J + covered_sum * dt;
    end
end
function tf = is_time_covered(t, rM, m, S, Psent, Pmed, Pfine, R_eff, vsink, eff_win)
    % 计算并收集此刻导弹M在场的所有云团数量
    C = zeros(0,3);
    for iU = 1:5
        if S.assign(iU) ~= m, continue; end
        for b = 1:3
            e = S.e(iU,b);
            if t < e || t > e + eff_win, continue; end
            c0 = squeeze(S.P_blow(iU,b,:)).';
            C(end+1,:) = c0 + [0,0,-(t-e)*vsink]; %#ok<AGROW>
        end
    end
    if isempty(C), tf = false; return; end

    % 每条连线用距最佳的一个云团中心来判断
    if ~cloudCoopCovers_fast(C, R_eff, rM, Psent), tf=false; return; end
    if ~cloudCoopCovers_fast(C, R_eff, rM, Pmed),  tf=false; return; end
    if ~cloudCoopCovers_fast(C, R_eff, rM, Pfine), tf=false; return; end
    tf = true;
end

function ok = cloudCoopCovers_fast(C, R_eff, rM, P)
    % 计算每条连线到所有云团中心的最近距离，并取最小值
    AB  = bsxfun(@minus, P, rM);       
    AB2 = sum(AB.^2, 2);               
    d2_min = inf(size(AB2));
    for j = 1:size(C,1)
        CA = C(j,:) - rM;              
        t  = zeros(size(AB2));
        nz = AB2 > 1e-12;
        t(nz) = (AB(nz,:)*CA.')./AB2(nz);
        t  = max(0, min(1, t));
        Q  = bsxfun(@plus, rM, bsxfun(@times, t, AB));   
        d2 = sum(bsxfun(@minus, C(j,:), Q).^2, 2);
        d2_min = min(d2_min, d2);
    end
    ok = all(d2_min <= (R_eff^2 + 1e-12));
end


function iv = active_windows(m, S, T, eff_win)
% 将导弹M的被遮蔽时间区间与时间的上下求交，再进行各自的求交
    segs = zeros(0,2);
    for i=1:5
        if S.assign(i)~=m, continue; end
        for b=1:3
            e = S.e(i,b);
            a = max(0, e); bnd = min(T, e + eff_win);
            if bnd > a, segs(end+1,:) = [a, bnd]; end %#ok<AGROW>
        end
    end
    if isempty(segs), iv = zeros(0,2); return; end
    segs = sortrows(segs,1);
    iv = segs(1,:);
    for k=2:size(segs,1)
        if segs(k,1) <= iv(end,2) + 1e-9
            iv(end,2) = max(iv(end,2), segs(k,2));
        else
            iv(end+1,:) = segs(k,:); %#ok<AGROW>
        end
    end
end

function detail = finalize_solution_nobis(gX, LB, UB, assign_map, UAV0, M0, uMO, vM, ...
    Psent, Pmed, Pfine, R_eff, vsink, eff_win, Tend, dt_fine, DEBUG)
% 编写函数，实现对最优解的函数的精评
    S = decode_Q5(gX, LB, UB, assign_map, UAV0);
    S = points_from_S(S, UAV0);
    Tsum=zeros(3,1); intervals=cell(3,1);
    for m=1:3
        iv = active_windows(m, S, Tend(m), eff_win);
        mask = false(0,1); ts_all = zeros(0,1);
        total_active_len = 0; total_samples = 0;
        for k=1:size(iv,1)
            a = iv(k,1); b = min(iv(k,2),Tend(m));
            total_active_len = total_active_len + max(0,b-a);
            ts = (a:dt_fine:b).';
            if isempty(ts), continue; end
            rM = bsxfun(@plus, M0(m,:), bsxfun(@times, vM*ts, uMO(m,:)));
            cov = false(numel(ts),1);
            for ti=1:numel(ts)
                cov(ti) = is_time_covered(ts(ti), rM(ti,:), m, S, Psent, Pmed, Pfine, R_eff, vsink, eff_win);
            end
            total_samples = total_samples + numel(ts);
            mask  = [mask;  cov]; 
            ts_all= [ts_all; ts];
        end
        Tsum(m) = sum(mask)*dt_fine;
        intervals{m} = mask_to_intervals(ts_all, mask);
        if DEBUG
            fprintf(' [M%d] active-ub=%.2fs, samples=%d, covered=%.3fs\n', ...
                m, total_active_len, total_samples, Tsum(m));
        end
    end
    if DEBUG
        fprintf(' ==> Sum over missiles: %.3f s\n', sum(Tsum));
    end
    detail = struct('S',S,'intervals',{intervals},'Tsum',Tsum,'Jsum',sum(Tsum));
end

function ivals = mask_to_intervals(ts, mask)
% 通过分散的掩码直接提取区间
    ts = ts(:); m = logical(mask(:));
    ivals = zeros(0,2);
    if isempty(ts) || isempty(m), return; end
    d = diff([false; m; false]);
    on  = find(d== 1);
    off = find(d==-1)-1;
    for i=1:numel(on)
        tL = ts(on(i));
        tR = ts(off(i));
        ivals(end+1,:) = [tL, tR];
    end
end

function S = decode_Q5(x, LB, UB, assign_map, UAV0)
%实现各种数据的归一化
    x = min(max(x,0),1);
    z = LB + (UB-LB).*x;
    psi=zeros(1,5); vF=zeros(1,5); d=zeros(5,3); D=zeros(5,3); e=zeros(5,3);
    for i=1:5
        idx=(i-1)*8;
        psi(i)=z(idx+1); vF(i)=z(idx+2);
        for b=1:3, d(i,b)=z(idx+2+2*b-1); D(i,b)=z(idx+2+2*b); end
        % 将数据进行成对排序
        [ds, ord] = sort(d(i,:));
        D_sorted  = D(i,ord);
        % 规定最小间隔
        for b=2:3, ds(b)=max(ds(b), ds(b-1)+1); end
        d(i,:)=ds; D(i,:)=D_sorted; e(i,:)=d(i,:)+D(i,:);
    end
    S=struct('psi',psi,'vF',vF,'d',d,'D',D,'e',e,'assign',assign_map);
end

function S = points_from_S(S, UAV0)
% 这个函数用于计算投掷点和爆炸点，并写入
    g=9.8;
    S.P_drop=zeros(5,3,3); S.P_blow=zeros(5,3,3);
    for i=1:5
        u=[cos(S.psi(i)), sin(S.psi(i)), 0];
        for b=1:3
            drop = UAV0(i,:) + S.vF(i)*S.d(i,b)*u; drop(3)=UAV0(i,3);
            tau = S.D(i,b);
            blow = drop + S.vF(i)*tau*u - 0.5*g*tau^2*[0,0,1];
            if blow(3)<0
                Delta_max = sqrt(max(2*drop(3)/g,0));
                tau = min(tau, Delta_max);
                S.D(i,b)=tau; S.e(i,b)=S.d(i,b)+tau;
                blow = drop + S.vF(i)*tau*u - 0.5*g*tau^2*[0,0,1];
            end
            S.P_drop(i,b,:) = drop; S.P_blow(i,b,:) = blow;
        end
    end
end

function ok = cloudFullyCovers_fast(c, R_eff, rM, P)
% 该函数可将部分数据实现向量化
    AB  = bsxfun(@minus, P, rM);           
    AB2 = sum(AB.^2, 2);                   
    CA  = c - rM;                          
    t   = zeros(size(AB2));
    nz  = AB2 > 1e-12;
    t(nz) = (AB(nz,:)*CA.')./AB2(nz);
    t   = max(0, min(1, t));
    Q   = bsxfun(@plus, rM, bsxfun(@times, t, AB));     % 近点
    d2  = sum(bsxfun(@minus, c, Q).^2, 2);
    if any(d2 > (R_eff^2 + 1e-12)), ok=false; return; end
    ok=true;
end

function P = sampleCylinderPoints(c0, R, H, side_th, side_z, cap_th, cap_r)
% 再圆柱体的表面实现采取若干给样
    th = linspace(0,2*pi, side_th+1); th(end)=[];
    z  = linspace(0,H, side_z); [TH,ZZ] = meshgrid(th,z);
    side = [R*cos(TH(:)), R*sin(TH(:)), ZZ(:)] + repmat(c0,size(TH(:),1),1);
    th2 = linspace(0,2*pi,cap_th+1); th2(end)=[];
    r2  = linspace(0,R,cap_r+1); r2(1)=[];
    [RR,TT] = meshgrid(r2, th2);
    capTop = [RR(:).*cos(TT(:)), RR(:).*sin(TT(:)), H*ones(numel(RR),1)] + repmat(c0,numel(RR),1);
    capBot = [RR(:).*cos(TT(:)), RR(:).*sin(TT(:)), zeros(numel(RR),1)] + repmat(c0,numel(RR),1);
    P = [side; capTop; capBot];
end

function seeds = heuristic_seeds(n, LB, UB, UAV0)
% 编写的启发式种子，有助于加快收敛使用
    D = numel(LB); seeds = rand(n,D);
    tgt = [0,200]; v_pref = 120;
    e_guess = [12, 22, 35];
    for i=1:min(5,n)
        idx=(i-1)*8;
        ang = atan2(tgt(2)-UAV0(i,2), tgt(1)-UAV0(i,1));
        seeds(i,idx+1) = (ang - LB(idx+1)) / (UB(idx+1)-LB(idx+1));  
        seeds(i,idx+2) = (v_pref - LB(idx+2)) / (UB(idx+2)-LB(idx+2)); 
        for b=1:3
            d_guess = max(0, e_guess(b) - 8 + 2*b);  
            D_guess = 8 + 2*b;
            seeds(i,idx+2+2*b-1) = (d_guess - LB(idx+2+2*b-1)) / (UB(idx+2+2*b-1)-LB(idx+2+2*b-1));
            seeds(i,idx+2+2*b  ) = (D_guess - LB(idx+2+2*b  )) / (UB(idx+2+2*b  )-LB(idx+2+2*b  ));
        end
    end
    seeds = min(max(seeds,0),1);
end
