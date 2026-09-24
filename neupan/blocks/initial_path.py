
"""
InitialPath is the class for generating the naive initial path for NeuPAN from the given waypoints.

Developed by Ruihua Han
Copyright (c) 2025 Ruihua Han

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
"""

import numpy as np
from math import tan, inf, cos, sin, sqrt
from gctl import curve_generator
from neupan.util import WrapToPi, distance
import math

class InitialPath:
    """
    generate initial path from the given waypoints
        waypoints: list of waypoints, waypoint: [x, y, yaw] or numpy array of shape (n, 3)
        loop: if True, the path and curve index will be reset to the beginning when reaching the end
    """

    def __init__(
        self,
        receding,
        step_time,
        ref_speed,
        robot,
        waypoints=None,
        loop=False,
        curve_style="line",
        **kwargs,
    ) -> None:

        self.T = receding
        self.dt = step_time
        self.ref_speed = ref_speed
        self.robot = robot
        self.waypoints = self.trans_to_np_list(waypoints)
        self.loop = loop
        self.curve_style = curve_style
        self.min_radius = kwargs.get("min_radius", self.default_turn_radius())
        self.interval = kwargs.get("interval", self.dt * self.ref_speed)
        self.arrive_threshold = kwargs.get("arrive_threshold", 0.1)
        self.close_threshold = kwargs.get("close_threshold", 0.1)
        self.ind_range = kwargs.get("ind_range", 10)
        self.arrive_index_threshold = kwargs.get("arrive_index_threshold", 1)
        self.arrive_flag = False

        # ---- 朝向参考从哪来 (只对 omni3 有意义) ------------------------------
        #   'path' -> ref_s 第三行取**全局初始路径**上参考点的方向 (原行为)。
        #   'plan' -> 取**规划器自己算出的轨迹**的行进方向, 即 /neupan_plan 的切向。
        #
        # 为什么需要 'plan': omni3 下 theta 确实是可控状态且进状态代价
        # (robot.C0_cost 的 diff_s 全三行), 但它跟踪的参考是全局路径的方向。
        # _ensure_consistent_angles 把路径每个点的 theta 设成"指向下一个**路径**
        # 点", 于是绕障时车体已经横移出去了, 朝向参考却还在说"对准全局直线",
        # 偏差约等于 0 -> w 约等于 0 -> 绕障全靠 vx/vy 蟹行。这就是实测现象
        # (避障过程中 yaw 几乎不变) 的原因, 不是 generate_twist_msg 漏发角速度。
        #
        # 'plan' 把参考换成规划器**当前这条轨迹**的行进方向, 车头就跟着绕障轨迹
        # 一起摆过去。
        #
        # 默认保持 'path': 这是 upstream 行为, 其它 example / limo 配置不受影响。
        self.heading_ref = str(kwargs.get("heading_ref", "path")).strip().lower()

        if self.heading_ref not in ("path", "plan"):
            raise ValueError(
                "ipath.heading_ref expects path|plan, got: %r" % self.heading_ref)

        # 'plan' 模式下, 规划速度模长低于这个值就认为方向不可辨识 (atan2 的结果
        # 基本是噪声), 退回该步的路径方向。终点附近速度自然衰减到 0, 于是朝向
        # 参考平滑地回到路径方向, 不会在停车时乱转。
        self.heading_speed_threshold = float(
            kwargs.get("heading_speed_threshold", 0.02))

        self.cg = curve_generator()
        # initial path and gear
        self.initial_path = None

        

    def generate_nom_ref_state(self, state: np.ndarray, cur_vel_array: np.ndarray, ref_speed: float):
        """
        state: current state of the robot, shape (3, 1)
        cur_vel_array: current velocity array of the robot, shape (2, T)
        """
        state = state[:3]

        ref_state = self.cur_point[0:3].copy()
        ref_index = self.point_index
        pre_state = state.copy()

        state_pre_list = [pre_state]
        state_ref_list = [ref_state]

        assert self.cur_point.shape[0] >= 4
        gear_list = [self.cur_point[-1, 0]] * self.T

        ref_speed_forward = ref_speed * self.dt

        # 'plan' 模式下每步的朝向参考 (世界系 rad), None 表示该步退回路径方向。
        plan_heading = self.plan_heading_list(cur_vel_array, gear_list)

        for t in range(self.T):
            pre_state = self.motion_predict_model(
                pre_state, cur_vel_array[:, t : t + 1], self.robot.L, self.dt
            )
            state_pre_list.append(pre_state)

            if ref_speed_forward >= self.interval:
                inc_index = int((ref_speed_forward) / self.interval)
                ref_index = ref_index + inc_index

                if ref_index > len(self.cur_curve) - 1:
                    ref_index = len(self.cur_curve) - 1
                    gear_list[t] = 0

                # astype(float) 同时修掉两个坑, 两个都只在 heading_ref='plan'
                # 下才会咬人, 所以原来一直是隐性的:
                #
                # 1) 必须是**副本**。原来取的是 cur_curve[ref_index][0:3] 的视图,
                #    而下面要写 ref_state[2, 0] —— 那等于直接改初始路径上那个点。
                #    原来侥幸无害 (写回的值与原值同余 2pi, 路径方向没变), 但
                #    'plan' 写进去的是规划轨迹方向, 会把全局路径的朝向逐点覆盖,
                #    一旦覆盖, 'path' 这条退路也就没了。
                # 2) 必须是**浮点**。gctl 生成的路径里个别点是整数 dtype (实测
                #    waypoints=[[0,0,0],[2,2,0],[4,0,0]] 下 idx 1/33/65 是 int64),
                #    往 int 数组写 0.785 rad 会被静默截断成 0, 表现为朝向参考
                #    "偶尔不生效"。'path' 下看不出来: 那几个点的 theta 本来就是 0。
                ref_state = self.cur_curve[ref_index][0:3].astype(float)

            else:
                ref_state, ref_index = self.find_interaction_point(
                    ref_state, ref_index, ref_speed_forward
                )

                if ref_index > len(self.cur_curve) - 1:
                    gear_list[t] = 0

            if plan_heading[t] is not None:
                ref_state[2, 0] = plan_heading[t]

            # 参考朝向按**预测状态**解卷绕。代价是二次的, 看不懂 ±pi 等价,
            # 所以必须把参考挪到离 pre_state 最近的那个同余值上, 否则车会为了
            # 消掉一个 2pi 的假偏差而绕远路转一圈。
            diff = ref_state[2, 0] - pre_state[2, 0]
            ref_state[2, 0] = pre_state[2, 0] + WrapToPi(diff)
            state_ref_list.append(ref_state)

        nom_s = np.hstack(state_pre_list)
        nom_u = cur_vel_array
        ref_s = np.hstack(state_ref_list)

        # if max(gear_list[1:]) < 0.001:
        #     gear_array = np.zeros(self.T)
        # else:
        gear_array = np.array(gear_list)

        ref_us = gear_array * ref_speed

        if self.robot.cartesian_vel:
            # omni/omni3 的控制量前两维是笛卡尔 (vx, vy), 代价函数要把速度分解到
            # 路径坐标系,
            # 所以额外给出每步的**单位切向** (2, T)。ref_us 仍是标量参考速度,
            # 含义不变 (沿路径方向的速度大小)。
            #
            # 切向用参考点的有限差分, 而不是 ref_s 第三行的 theta —— 对全向底盘
            # theta 是车头朝向, 跟运动方向无关, 而且上面刚被 WrapToPi 改写过。
            ref_xy = ref_s[0:2, :]                      # (2, T+1)
            tangent = ref_xy[:, 1:] - ref_xy[:, :-1]    # (2, T)
            norm = np.linalg.norm(tangent, axis=0, keepdims=True)
            # 路径末端相邻参考点重合 -> 切向为零。此时 gear 也已置 0, ref_us 是 0,
            # 沿路径分量的目标就是 0, 与"到点停住"一致。切向填成 (1,0) 只是避免
            # 除零, 乘上 ref_us=0 后不影响结果。
            degenerate = norm < 1e-9
            safe = np.where(degenerate, 1.0, norm)
            unit_tangent = tangent / safe
            unit_tangent[:, degenerate[0]] = np.array([[1.0], [0.0]])
            return nom_s, nom_u, ref_s, ref_us, unit_tangent

        return nom_s, nom_u, ref_s, ref_us

    def plan_heading_list(self, cur_vel_array, gear_list):

        '''
        heading_ref='plan' 时每步的朝向参考: **规划轨迹的行进方向**。

        参考取自 cur_vel_array —— 上一帧 NRMP 解出的最优速度序列, 也就是这一帧
        SCP 的展开点 (nom_u), 同时正是 /neupan_plan 那条轨迹对应的速度。omni3 的
        动力学 x_{t+1} = x_t + v_t*dt 是**精确**的 (linear_omni3_model 的 A=I,
        B=dt*I, C=0), 所以 atan2(vy_t, vx_t) 与 opt_state 相邻两点之差的方向严格
        相等, 直接用速度比对位置做有限差分更干净, 也不用担心末端重合点除零。

        只对 omni3 生效:
          - diff/acker 的 cur_vel_array 是 (v, w) / (v, psi), 前两行**不是**
            笛卡尔速度, atan2 出来是无意义的数 —— 必须挡住, 否则是静默的错。
            这两种底盘也不需要: 非完整约束下车头本来就等于行进方向。
          - omni (2 自由度) 的 theta 结构上不可控 (B 第三行恒 [0,0]),
            robot.C0_cost 里 diff_s 只算 [0:2], 朝向参考写什么都不起作用。

        返回长度 T 的列表, 元素是 float (该步的朝向参考) 或 None (退回路径方向)。
        '''

        none_list = [None] * self.T

        if self.heading_ref != "plan":
            return none_list

        if not (self.robot.cartesian_vel and self.robot.yaw_controllable):
            return none_list

        headings = []

        for t in range(self.T):
            vx, vy = cur_vel_array[0, t], cur_vel_array[1, t]

            # 速度太小时方向不可辨识。终点附近 (gear 置 0, ref_us=0) 速度自然
            # 衰减到 0, 于是朝向参考平滑退回路径方向, 停车时不会原地乱转。
            # 第一帧 cur_vel_array 是全零, 这里也会整列返回 None, 车按路径方向
            # 起步 —— 与改动前的行为一致。
            if np.hypot(vx, vy) < self.heading_speed_threshold:
                headings.append(None)
            else:
                headings.append(float(np.arctan2(vy, vx)))

        return headings

    def set_initial_path(self, path):

        '''
        set the initial path from the given path

        Args:
            path: list of points, each point is a numpy array of shape (4, 1)
        '''

        self.initial_path = path
        self.interval = self.cal_average_interval(path)
        self.split_path_with_gear()
        self.curve_index = 0
        self.point_index = 0


    def cal_average_interval(self, path):

        '''
        calculate the average interval of the given path

        Args:
            path: list of points, each point is a numpy array of shape (4, 1)
        '''

        n = len(path)

        if n < 2:
            return 0
        
        dist_sum = 0.0
        for point1, point2 in zip(path, path[1:]):
            x1, y1 = point1[0:2]
            x2, y2 = point2[0:2]
            dist_sum += math.hypot(x2 - x1, y2 - y1)

        return dist_sum / (n - 1)

    def closest_point(self, state, threshold=0.1, ind_range=10):

        min_dis = inf
        cur_index = self.point_index

        start = max(cur_index, 0)
        end = min(cur_index + ind_range, len(self.cur_curve))

        for index in range(start, end):
            dis = distance(state[0:2], self.cur_curve[index][0:2])

            if dis < min_dis:
                min_dis = dis
                self.point_index = index
                if dis < threshold:
                    break

        return min_dis

    def find_interaction_point(self, ref_state, ref_index, length):

        circle = np.squeeze(ref_state[0:2])

        while True:

            if ref_index > len(self.cur_curve) - 2:
                # .copy() 同 generate_nom_ref_state 里那处: 原来返回的是
                # cur_curve[-1] 的视图, 调用方要写第三行, 等于改初始路径末点。
                # heading_ref='plan' 下写进去的是规划轨迹方向, 会真的破坏路径。
                end_point = self.cur_curve[-1].astype(float)
                end_point[2] = WrapToPi(end_point[2])

                return end_point[0:3], ref_index

            cur_point = self.cur_curve[ref_index]
            next_point = self.cur_curve[ref_index + 1]
            segment = [np.squeeze(cur_point[0:2]), np.squeeze(next_point[0:2])]
            interaction_point = self.range_cir_seg(circle, length, segment)

            if interaction_point is not None:
                diff = WrapToPi(next_point[2, 0] - cur_point[2, 0])
                theta = WrapToPi(cur_point[2, 0] + diff / 2)
                state_ref = np.append(interaction_point, theta).reshape((3, 1))

                return state_ref, ref_index

            else:
                ref_index += 1

    def range_cir_seg(self, circle, r, segment):

        # find the intersection point between the circle and the segment

        assert (
            circle.shape == (2,)
            and segment[0].shape == (2,)
            and segment[1].shape == (2,)
        )

        sp = segment[0]
        ep = segment[1]

        d = ep - sp

        if np.linalg.norm(d) == 0:
            return None

        f = sp - circle

        a = d @ d
        b = 2 * f @ d
        c = f @ f - r**2

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return None
        else:

            # t1 = (-b - sqrt(discriminant)) / (2 * a)
            t2 = (-b + sqrt(discriminant)) / (2 * a)

            if t2 >= 0 and t2 <= 1:
                int_point = sp + t2 * d
                return int_point

            return None

    def check_arrive(
        self,
        state,
    ):

        self.init_check(state)  # check if the initial path is set
        self.closest_point(
            state, self.close_threshold, self.ind_range
        )  # find the closest point on the path

        if self.check_curve_arrive(state, self.arrive_threshold, self.arrive_index_threshold):
            
            if self.curve_index + 1 >= self.curve_number:
                
                if self.loop:
                    self.curve_index = 0
                    self.point_index = 0

                    print("Info: loop, reset the path")
                    # self.initial_path.reverse()
                    # self.split_path_with_gear()
                    return False
                else:
                    if not self.arrive_flag:
                        print("Info: arrive at the end of the path")
                        self.arrive_flag = True
                    return True
            else:
                self.curve_index += 1
                self.point_index = 0

        return False

    def check_curve_arrive(self, state, arrive_threshold=0.1, arrive_index_threshold=2):

        final_point = self.cur_curve[-1][0:2]
        arrive_distance = np.linalg.norm(state[0:2] - final_point)

        return(
            arrive_distance < arrive_threshold
            and self.point_index >= (len(self.cur_curve) - arrive_index_threshold - 2)
        )

    def split_path_with_gear(self):
        """
        split initial path into multiple curves by gear
        """

        if not hasattr(self, "initial_path"):
            raise AttributeError("Object must have a 'initial_path' attribute")

        self.curve_list = []
        current_curve = []
        current_gear = self.initial_path[0][-1]

        for point in self.initial_path:
            if point[-1] != current_gear:
                self.curve_list.append(current_curve)
                current_curve = []
                current_gear = point[-1]

            current_curve.append(point)

        # Append the last curve
        if current_curve:
            self.curve_list.append(current_curve)

    def init_path_with_state(self, state):

        assert len(self.waypoints) > 0, "Error: waypoints are not set"

        if isinstance(self.waypoints, list):
            self.waypoints = [state] + self.waypoints
        elif isinstance(self.waypoints, np.ndarray):
            self.waypoints = np.vstack([state, self.waypoints])

        if self.loop:
            self.waypoints = self.waypoints + [self.waypoints[0]]

        self.initial_path = self.cg.generate_curve(
            self.curve_style, self.waypoints, self.interval, self.min_radius, True
        )

        if self.curve_style == 'line':
            # Ensure consistent angles for line curve
            self._ensure_consistent_angles()

    def init_check(self, state):

        if self.initial_path is None:
            print("initial path is not set, generate path with the current state")
            self.set_ipath_with_state(state)

    def set_ipath_with_state(self, state):

        self.init_path_with_state(state[0:3])
        self.split_path_with_gear()
        # self.path_index = 0
        self.curve_index = 0
        self.point_index = 0

    def update_initial_path_from_goal(self, start, goal):

        if self.loop:
            waypoints = [start, goal, start]
        else:
            waypoints = [start, goal]

        self.initial_path = self.cg.generate_curve(
            self.curve_style, waypoints, self.interval, self.min_radius, True
        )
        
        if self.curve_style == 'line':
            # Ensure consistent angles for line curve
            self._ensure_consistent_angles()

        self.split_path_with_gear()
        # self.path_index = 0
        self.curve_index = 0
        self.point_index = 0
        self.waypoints = waypoints

    def set_ipath_with_waypoints(self, waypoints):
        self.initial_path = self.cg.generate_curve(
            self.curve_style, waypoints, self.interval, self.min_radius, True
        )
        
        if self.curve_style == 'line':
            # Ensure consistent angles for line curve
            self._ensure_consistent_angles()

        self.split_path_with_gear()
        # self.path_index = 0
        self.curve_index = 0
        self.point_index = 0
        self.waypoints = waypoints

    def motion_predict_model(self, robot_state, vel, wheel_base, sample_time):

        if self.robot.kinematics == "acker":
            next_state = self.ackermann_model(robot_state, vel, wheel_base, sample_time)

        elif self.robot.kinematics == "diff":
            next_state = self.diff_model(robot_state, vel, sample_time)
        
        elif self.robot.kinematics == "omni":
            next_state = self.omni_model(robot_state, vel, sample_time)

        elif self.robot.kinematics == "omni3":
            next_state = self.omni3_model(robot_state, vel, sample_time)

        else:
            raise ValueError(
                "unsupported kinematics: %r" % self.robot.kinematics)

        return next_state

    def ackermann_model(self, car_state, vel, wheel_base, sample_time):

        assert car_state.shape == (3, 1) and vel.shape == (2, 1)

        phi = car_state[2, 0]

        v = vel[0, 0]
        psi = vel[1, 0]

        ds = np.array([[v * cos(phi)], [v * sin(phi)], [v * tan(psi) / wheel_base]])

        next_state = car_state + ds * sample_time

        # next_state[2, 0] = wraptopi(next_state[2, 0])

        return next_state

    def diff_model(self, robot_state, vel, sample_time):

        assert robot_state.shape == (3, 1) and vel.shape == (2, 1)

        phi = robot_state[2, 0]
        v = vel[0, 0]
        w = vel[1, 0]

        ds = np.array([[v * cos(phi)], [v * sin(phi)], [w]])

        next_state = robot_state + ds * sample_time

        # next_state[2, 0] = wraptopi(next_state[2, 0])

        return next_state
    
    def omni_model(self, robot_state, vel, sample_time):

        # vel 是**世界系**笛卡尔速度 (vx, vy), 与 robot.linear_omni_model 一致。
        assert robot_state.shape[0] >= 2 and vel.shape == (2, 1)

        omni_vel = np.array([[vel[0, 0]], [vel[1, 0]], [0]])

        next_state = robot_state + sample_time * omni_vel

        return next_state

    def omni3_model(self, robot_state, vel, sample_time):

        # vel = (vx, vy, w)。vx/vy 是**世界系**笛卡尔平移速度, w 是角速度,
        # 与 robot.linear_omni3_model 一致 —— 这里是那个线性模型的直接积分,
        # 所以两者**必须**保持一致, 否则 nom_s 和凸问题里的动力学约束会错位。
        assert robot_state.shape[0] >= 3 and vel.shape == (3, 1)

        omni_vel = np.array([[vel[0, 0]], [vel[1, 0]], [vel[2, 0]]])

        next_state = robot_state + sample_time * omni_vel

        return next_state

    @property
    def cur_waypoints(self):
        return self.waypoints

    @property
    def cur_curve(self):
        return self.curve_list[self.curve_index]

    @property
    def cur_point(self):
        return self.cur_curve[self.point_index]

    @property
    def curve_number(self):
        return len(self.curve_list)

    def default_turn_radius(self):

        if self.robot.kinematics == "acker":
            max_psi = self.robot.max_speed[1]
            default_radius = self.robot.L / tan(max_psi)  # radius =  wheelbase / tan(psi)
        else:
            default_radius = 0.0

        return default_radius

    def _ensure_consistent_angles(self):
        """
        Ensure that all points in the initial path have consistent angles.
        For line curves, angles should represent the direction of travel.
        """
        if self.initial_path is None or len(self.initial_path) < 2:
            return
        
        for i in range(len(self.initial_path) - 1):
            current_point = self.initial_path[i]
            next_point = self.initial_path[i + 1]
            
            dx = next_point[0, 0] - current_point[0, 0]
            dy = next_point[1, 0] - current_point[1, 0]
            
            theta = math.atan2(dy, dx)
            
            current_point[2, 0] = theta
        
        if len(self.initial_path) >= 2:
            self.initial_path[-1][2, 0] = self.initial_path[-2][2, 0]

    def trans_to_np_list(self, point_list):

        if point_list is None:
            return []

        return [np.c_[p] if isinstance(p, list) else p for p in point_list]