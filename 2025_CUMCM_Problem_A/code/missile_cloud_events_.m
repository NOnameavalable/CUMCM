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
