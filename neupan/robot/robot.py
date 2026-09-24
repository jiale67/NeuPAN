'''
robot class define the robot model and the kinematics model for NeuPAN. It also generate the constraints and cost functions for the optimization problem.

Developed by Ruihua Han
Copyright (c) 2025 Ruihua Han <hanrh@connect.hku.hk>

NeuPAN planner is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

NeuPAN planner is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with NeuPAN planner. If not, see <https://www.gnu.org/licenses/>.
'''

# Python 3.8 兼容：延迟注解求值（PEP 563），详见 neupan/util/__init__.py
from __future__ import annotations

from math import inf
import numpy as np
from typing import Optional, Union
import cvxpy as cp
from math import sin, cos, tan
import torch
from neupan.configuration import to_device 
from neupan.util import gen_inequal_from_vertex

class robot:

    def __init__(
        self,
        receding: int = 10,
        step_time: float = 0.1,
        kinematics: Optional[str] = None,
        vertices: Optional[Union[list[float], np.ndarray]] = None,
        max_speed: list[float] = [inf, inf],
        max_acce: list[float] = [inf, inf],
        wheelbase: Optional[float] = None,
        length: Optional[float] = None,
        width: Optional[float] = None,
        **kwargs,
    ):
        
        if kinematics is None:
            raise ValueError("kinematics is required")

        self.shape = None

        self.vertices = self.cal_vertices(vertices, length, width, wheelbase)
        
        self.G, self.h = gen_inequal_from_vertex(self.vertices)

        self.T = receding
        self.dt = step_time
        self.L = wheelbase

        self.kinematics = kinematics
        self.max_speed = np.c_[max_speed] if isinstance(max_speed, list) else max_speed
        self.max_acce = np.c_[max_acce] if isinstance(max_acce, list) else max_acce

        if kinematics == 'acker':
            if self.max_speed[1] >= 1.57:
                print(f"Warning: max steering angle of acker robot is {self.max_speed[1]} rad, which is larger than 1.57 rad, so it is limited to 1.57 rad")
                self.max_speed[1] = 1.57

        self.speed_bound = self.max_speed
        self.acce_bound = self.max_acce * self.dt

        # omni 专用: 垂直于路径方向的速度分量, 代价权重相对 p_u 的**比例**
        # (见 C0_cost)。用比例而不是绝对值, 因为起作用的是二者之比 ——
        # 同一个绝对值在 p_u=2.5 和 p_u=1.0 下含义完全不同。
        #
        #   1.0  各向同性。此时代价恰好等于"整个速度矢量偏离参考速度矢量"
        #        (旋转不改变 2-范数), 方向和大小都被钉住。空场景跟踪最准,
        #        但横移要付全价, 障碍物当前时倾向减速而不是绕开 —— 实测
        #        simple_S1 绕行要 57.2s。
        #   0.2  横移比前进便宜 5 倍。实测 36.7s, v_along 稳在 0.303,
        #        绕行横向偏离 0.37m, 轨迹离 box_16 最近 0.28m。
        #   0.0  横向完全免费。避障灵但横向失控: 障碍物附近速度方向 ±90 度
        #        跳变, |v| 冲到 0.60 (ref_speed 只有 0.3); 无障碍时也会横向
        #        漂移 (NeuPAN non_obs/omni example 弧长比 1.13)。
        #
        # 默认 1.0: 各向同性是最保守的选择, 且与"直接惩罚速度矢量"完全等价。
        self.lateral_ratio = float(kwargs.get("lateral_ratio", 1.0))

        self.name = kwargs.get("name", self.kinematics + "_robot" + '_default') 

    def define_variable(self, no_obs: bool = False, indep_dis: cp.Variable = None):

        """
        define variables
        """

        self.indep_s = cp.Variable((3, self.T + 1), name="state")  # t0 - T
        self.indep_u = cp.Variable((2, self.T), name="vel")  # t1 - T

        if self.kinematics == 'omni':
            # 速度沿路径方向的投影。本来可以直接写
            #   para_p_u * sum(para_gamma_tangent * indep_u)
            # 但那是 参数 x 参数 x 变量, 不满足 DPP, CvxpyLayer 会拒绝。
            # 引入这个辅助变量后, 约束 indep_u_along == sum(tangent * u) 是
            # 参数仿射的, 代价里 para_p_u * indep_u_along 也只含一个参数,
            # DPP 成立, 同时 p_u 仍是可微可调的 Parameter (adjust 接口不变)。
            self.indep_u_along = cp.Variable((self.T,), name="vel_along_path")
            # 垂直于路径的分量。法向 = 切向逆时针转 90 度, 由同一个切向参数导出,
            # 不需要额外参数。
            self.indep_u_perp = cp.Variable((self.T,), name="vel_perp_path")

        indep_list = (
            [self.indep_s, self.indep_u]
            if no_obs
            else [self.indep_s, self.indep_u, indep_dis]
        )

        return indep_list

    def state_parameter_define(self):

        '''
        state parameters:
            - para_gamma_a: q*reference state, 3 * (T+1)
            - para_gamma_b: p*reference speed array, T
            - para_s: nominal state, 3 * (T+1)
            - para_A_list, para_B_list, para_C_list: for state transition model
        '''

        self.para_s = cp.Parameter((3, self.T+1), name='para_state')
        self.para_gamma_a = cp.Parameter((3, self.T+1), name='para_gamma_a')

        self.para_gamma_b = cp.Parameter((self.T,), name='para_gamma_b')

        if self.kinematics == 'omni':
            # omni 额外需要路径切向 (单位矢量), 用来把速度分解成沿路径/垂直路径
            # 两个分量。只惩罚沿路径分量, 垂直分量留给避障 (见 C0_cost 的说明)。
            self.para_gamma_tangent = cp.Parameter((2, self.T), name='para_gamma_tangent')

        self.para_A_list = [ cp.Parameter((3, 3), name='para_A_'+str(t)) for t in range(self.T)]
        self.para_B_list = [ cp.Parameter((3, 2), name='para_B_'+str(t)) for t in range(self.T)]
        self.para_C_list = [ cp.Parameter((3, 1), name='para_C_'+str(t)) for t in range(self.T)]

        para_list = [self.para_s, self.para_gamma_a, self.para_gamma_b]

        if self.kinematics == 'omni':
            para_list += [self.para_gamma_tangent]

        return para_list + self.para_A_list + self.para_B_list + self.para_C_list


    def coefficient_parameter_define(self, no_obs: bool = False, max_num: int = 10):

        """
        gamma_c: lam.T
        zeta_a: lam.T @ p + mu.T @ h
        """

        if no_obs:
            self.para_gamma_c, self.para_zeta_a = [], []

        else:
            self.para_gamma_c = [
                cp.Parameter(
                    (max_num, 2),
                    value=np.zeros((max_num, 2)),
                    name="para_gamma_c" + str(i),
                )
                for i in range(self.T)
            ]  # lam.T, fa
            self.para_zeta_a = [
                cp.Parameter(
                    (max_num, 1),
                    value=np.zeros((max_num, 1)),
                    name="para_zeta_a" + str(i),
                )
                for i in range(self.T)
            ]  # lam.T @ p + mu.T @ h, fb

        return self.para_gamma_c + self.para_zeta_a


    def C0_cost(self, para_p_u, para_q_s):

        '''
        reference state cost and control vector cost

        para_p_u: weight of speed cost
        para_q_s: weight of state cost (scalar or 3-element vector for x, y, theta)
        '''

        if self.kinematics == 'omni':
            # 笛卡尔控制量 u = (vx, vy)。把速度**分解**到路径坐标系:
            #
            #   沿路径分量 u_along = tangent . u    -> 惩罚它偏离 ref_speed
            #   垂直路径分量 u_perp = normal . u    -> 按 lateral_ratio 打折惩罚
            #
            # 为什么横向要**打折**而不是同价 (lateral_ratio=1.0, 等价于直接惩罚
            # 整个速度矢量偏离 tangent * ref_speed): 同价时横移要付
            # p_u^2 * v_perp^2, 而横移对"沿路径前进"这一项毫无贡献, 于是障碍物
            # 正前方时减速比绕开便宜, 极端情况下 QP 的最优解是**停在安全边界上**
            # (I_cost 此时恰好为 0, 是个稳定局部极小)。实测三种墙 (对称/偏置/短墙)
            # 全部卡死在 d_max 边界, vy 最大 0.0002。
            #
            # 为什么不干脆**不惩罚** (lateral_ratio=0.0): 自由就是不受控。横向位置
            # 靠 diff_s 的 x,y 项拉回路径, 但横向**速度**没有任何约束, 障碍物附近
            # QP 会把速度全灌到横向, 只剩 norm(u) 限幅拦着 —— 实测 |v| 冲到 0.60
            # (ref_speed 只有 0.3), 速度方向 ±90 度跳变。
            #
            # 注: 极坐标版本之所以"能避障", 靠的正是 phi 完全不受惩罚 —— 那既是
            # 抖动的根源, 也是避障的唯一机制。这里把两者拆开: 方向自由度保留
            # (按比例定价, 而不是二选一), 且不再经过一个需要线性化的非线性模型。
            #
            # indep_u_along / indep_u_perp 由 bound_su_constraints() 约束为
            # tangent . u 和 normal . u。
            diff_u = para_p_u * self.indep_u_along - self.para_gamma_b

            # 横向分量单独一项, 权重 = lateral_ratio * p_u。常数 x 参数 x 变量
            # 仍只含一个参数, DPP 成立。
            # lateral_ratio=1.0 时这两项合起来恰好等于
            #   sum_squares(p_u * u - p_u * tangent * ref_us)
            # 即"整个速度矢量偏离参考速度矢量", 因为旋转到路径坐标系不改变范数。
            if self.lateral_ratio > 0.0:
                diff_u = cp.hstack(
                    [diff_u, (self.lateral_ratio * para_p_u) * self.indep_u_perp])
        else:
            # diff/acker: u[0] 是标量前向速度, u[1] 是角速度/前轮转角, 不惩罚。
            diff_u = para_p_u * self.indep_u[0, :] - self.para_gamma_b

        # Support both scalar and vector para_q_s
        # If para_q_s is a vector (3,1), use element-wise multiplication
        # If para_q_s is a scalar, use scalar multiplication
        if para_q_s.shape == (3, 1):
            # Element-wise multiplication: para_q_s (3,) broadcasts with indep_s (3, T+1)
            diff_s = cp.multiply(para_q_s, self.indep_s) - self.para_gamma_a
        else:
            # Scalar multiplication (backward compatibility)
            diff_s = para_q_s * self.indep_s - self.para_gamma_a

        if self.kinematics == 'omni':
            diff_s_cost = cp.sum_squares(diff_s[0:2])
        else:
            diff_s_cost = cp.sum_squares(diff_s)

        C0_cost = diff_s_cost + cp.sum_squares(diff_u)

        return C0_cost

    def proximal_cost(self):

        """
        proximal cost
        """

        proximal_cost = cp.sum_squares(self.indep_s - self.para_s)

        return proximal_cost


    def I_cost(self, indep_dis, ro_obs):

        cost = 0
        indep_t = self.indep_s[0:2, 1:]

        I_list = []

        for t in range(self.T):

            I_dpp = self.para_gamma_c[t] @ indep_t[:, t:t+1] - self.para_zeta_a[t] - indep_dis[0, t]
            I_list.append(I_dpp)

        I_array = cp.vstack(I_list)
        cost += 0.5 * ro_obs * cp.sum_squares(cp.neg(I_array))

        return cost

    def dynamics_constraint(self):

        '''
        linear dynamics constraints: x_{t+1} = A_t @ x_t + B_t @ u_t + C_t
        '''

        temp_list = []

        for t in range(self.T):
            indep_st = self.indep_s[:, t:t+1]
            indep_ut = self.indep_u[:, t:t+1]

            ## dynamic constraints
            A = self.para_A_list[t]
            B = self.para_B_list[t]
            C = self.para_C_list[t]
            
            temp_list.append(A @ indep_st + B @ indep_ut + C)
        
        constraints = [ self.indep_s[:, 1:] == cp.hstack(temp_list) ]

        return constraints 


    def bound_su_constraints(self):

        '''
        bound constraints on init state, speed, and acceleration   
        '''

        constraints = []

        if self.kinematics == 'omni':
            # 笛卡尔 (vx, vy) 下用二阶锥限幅而不是箱式限幅: cp.abs(u) <= bound
            # 是个方形包络, 对角方向会放行 sqrt(2)*max_speed。norm 限的是真实
            # 对地速度/加速度大小, 各向同性, 也才是麦轮该有的物理含义。
            # max_speed[0] / max_acce[0] 是速度和加速度的**模长**上限,
            # 第二项对 omni 不再使用。
            constraints += [ cp.norm(self.indep_u[:, 1:] - self.indep_u[:, :-1], axis=0)
                             <= float(self.acce_bound[0, 0]) ]
            constraints += [ cp.norm(self.indep_u, axis=0) <= float(self.speed_bound[0, 0]) ]
        else:
            constraints += [ cp.abs(self.indep_u[:, 1:] - self.indep_u[:, :-1] ) <= self.acce_bound ]
            constraints += [ cp.abs(self.indep_u) <= self.speed_bound]

        constraints += [ self.indep_s[:, 0:1] == self.para_s[:, 0:1] ]

        if self.kinematics == 'omni':
            # 定义沿路径/垂直路径两个分量。都是参数仿射, DPP 安全。
            #   tangent = (tx, ty)          -> along = tx*vx + ty*vy
            #   normal  = (-ty, tx)         -> perp  = -ty*vx + tx*vy
            tang = self.para_gamma_tangent
            normal = cp.vstack([-tang[1, :], tang[0, :]])

            constraints += [ self.indep_u_along
                             == cp.sum(cp.multiply(tang, self.indep_u), axis=0) ]
            constraints += [ self.indep_u_perp
                             == cp.sum(cp.multiply(normal, self.indep_u), axis=0) ]

        return constraints
    

    def generate_state_parameter_value(self, nom_s, nom_u, qs_ref_s, pu_ref_us,
                                       ref_tangent=None):

        '''
        ref_tangent: 仅 omni 需要, 单位路径切向 (2, T), 不含 p_u。
                     顺序必须与 state_parameter_define 返回的列表一致。
        '''

        state_value_list = [nom_s, qs_ref_s, pu_ref_us]

        if self.kinematics == 'omni':
            if ref_tangent is None:
                raise ValueError(
                    "omni kinematics requires ref_tangent (unit path tangent)")
            state_value_list = state_value_list + [ref_tangent]

        tensor_A_list = []
        tensor_B_list = []
        tensor_C_list = []

        for t in range(self.T):
            nom_st = nom_s[:, t:t+1]
            nom_ut = nom_u[:, t:t+1]

            if self.kinematics == 'acker':
                A, B, C = self.linear_ackermann_model(nom_st, nom_ut, self.dt, self.L)
            elif self.kinematics == 'diff':
                A, B, C = self.linear_diff_model(nom_st, nom_ut, self.dt)
            elif self.kinematics == 'omni':
                A, B, C = self.linear_omni_model(nom_ut, self.dt)
            else:
                raise ValueError('kinematics currently only supports acker or diff')

            tensor_A_list.append(A)
            tensor_B_list.append(B)
            tensor_C_list.append(C)

        state_value_list += tensor_A_list
        state_value_list += tensor_B_list
        state_value_list += tensor_C_list

        return state_value_list


    
    def linear_ackermann_model(self, nom_st, nom_ut, dt, L):
        
        phi = nom_st[2, 0]
        v, psi = nom_ut[0, 0], nom_ut[1, 0]

        A = torch.Tensor([ [1, 0, -v * dt * sin(phi)], [0, 1, v * dt * cos(phi)], [0, 0, 1] ])

        B = torch.Tensor([ [cos(phi)*dt, 0], [sin(phi)*dt, 0], 
                        [ tan(psi)*dt / L, v*dt/(L * (cos(psi))**2 ) ] ])

        C = torch.Tensor([ [ phi*v*sin(phi)*dt ], [ -phi*v*cos(phi)*dt ], 
                        [ -psi * v*dt / ( L * (cos(psi))**2) ]])
        

        return to_device(A), to_device(B), to_device(C)   
    

    def linear_diff_model(self, nom_state, nom_u, dt):
        
        phi = nom_state[2, 0]
        v = nom_u[0, 0]

        A = torch.Tensor([ [1, 0, -v * dt * sin(phi)], [0, 1, v * dt * cos(phi)], [0, 0, 1] ])

        B = torch.Tensor([ [cos(phi)*dt, 0], [sin(phi)*dt, 0], 
                        [ 0, dt ] ])

        C = torch.Tensor([ [ phi*v*sin(phi)*dt ], [ -phi*v*cos(phi)*dt ], 
                        [ 0 ]])
                
        return to_device(A), to_device(B), to_device(C) 
    
    def linear_omni_model(self, nom_u, dt):

        '''
        全向底盘模型。控制量是**世界系**笛卡尔速度 u = (vx, vy)。

            x_{t+1} = x_t + vx_t * dt
            y_{t+1} = y_t + vy_t * dt
            theta_{t+1} = theta_t

        这个模型是**精确线性**的: A/B/C 都是常数, 不依赖展开点 nom_u, 所以
        SCP 外层迭代不引入任何线性化误差。

        历史说明: 原实现用极坐标 u = (v, phi), phi 是方向角, 于是 B/C 依赖
        展开点, 需要线性化。那个线性化有个病态 —— 线性位移的幅值随 dphi 单调
        增大 (dphi=54deg 时 +37.4%), 也就是模型误以为"把方向角转开就能走得更
        远", 而横向分量 dt*v_n*dphi 既不消耗前向速度也不进代价函数。QP 会主动
        利用这个假象, 把 phi 顶到信赖域边界, 执行后过冲、反号, 形成约 1Hz 的
        极限环 (实测方向角 std 22.9deg, 弧长比 1.11)。极坐标还额外带来 v=0 处
        phi 不可辨识 (B 的 phi 列恒为零向量) 和 ±pi 处无角度归一化两个问题。
        笛卡尔参数化把这些一并消掉, 且下游本来就要 (vx, vy)。

        nom_u 保留在签名里只为与 diff/acker 调用形式一致, 本模型不使用它。
        '''

        A = torch.Tensor([ [1, 0, 0], [0, 1, 0], [0, 0, 1] ])
        B = torch.Tensor([ [ dt, 0 ], [ 0, dt ], [ 0, 0 ] ])
        C = torch.zeros((3, 1))

        return to_device(A), to_device(B), to_device(C)

    def cal_vertices_from_length_width(self, length, width, wheelbase=None):
        """
        Calculate initial vertices of a rectangle representing a robot.

        Args:
            length (float): Length of the rectangle.
            width (float): Width of the rectangle.
            wheelbase (float): Wheelbase of the robot.

        Returns:
            vertices (np.ndarray): Vertices of the rectangle. shape: (2, 4)
        """
        wheelbase = 0 if wheelbase is None else wheelbase

        start_x = -(length - wheelbase) / 2
        start_y = -width / 2

        point0 = np.array([[start_x], [start_y]])  # left bottom point
        point1 = np.array([[start_x + length], [start_y]])
        point2 = np.array([[start_x + length], [start_y + width]])
        point3 = np.array([[start_x], [start_y + width]])

        return np.hstack((point0, point1, point2, point3))
    
    def cal_vertices(self, vertices = None, length = None, width = None, wheelbase=None):

        '''
        Generate vertices. If vertices is not set, generate vertices from length, width, and wheelbase.

        Args:
            vertices: list of vertices or numpy array of vertices, [[x1, y1], [x2, y2], ...] or (2, N)
            length: length of the robot
            width: width of the robot
            wheelbase: wheelbase of the robot

        Returns:
            vertices_np: numpy array of vertices, shape: (2, N), N >3
        '''

        if vertices is not None:
           if isinstance(vertices, list):
                vertices_np = np.array(vertices).T

           elif isinstance(vertices, np.ndarray):
                vertices_np = vertices
           else:
                raise ValueError("vertices must be a list or numpy array")
           
        else:
            self.shape = "rectangle"
            vertices_np = self.cal_vertices_from_length_width(length, width, wheelbase)
            self.length = length
            self.width = width
            self.wheelbase = wheelbase

        assert vertices_np.shape[1] >= 3, "vertices must be a numpy array of shape (2, N), N >= 3"

        return vertices_np

