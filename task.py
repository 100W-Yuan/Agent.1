class Task:

    def __init__(self, user_message: str):

        # 用户提出的任务
        self.user_message = user_message

        # 任务状态
        self.status = "pending"

        # 任务计划
        self.plan = []

        # 实际执行记录
        self.steps = []

    def start(self):
        """开始执行任务。"""
        self.status = "running"

    def add_plan_step(
        self,
        description: str,
        step_type: str = "reasoning",
        tool_name: str | None = None
    ):
        """
        添加一个计划步骤。

        step_type:
        - tool: 需要调用工具
        - reasoning: Agent 自己进行信息处理、判断或生成结果

        tool_name:
        - 当 step_type == "tool" 时，可以指定必须调用的工具名称。
        - reasoning 步骤通常为 None。
        """

        self.plan.append({
            "description": description,
            "type": step_type,
            "tool_name": tool_name,
            "status": "pending"
        })

    def complete_plan_step(self, description: str):
        """把一个计划步骤标记为完成。"""

        for step in self.plan:

            if step["description"] == description:

                step["status"] = "completed"

                return

    def fail_plan_step(self, description: str):
        """把一个计划步骤标记为失败。"""

        for step in self.plan:

            if step["description"] == description:

                step["status"] = "failed"

                return

    def add_step(
        self,
        description: str,
        status: str = "completed",
        plan_steps: list[int] | None = None
    ):
        """记录一个实际执行步骤。"""

        self.steps.append({
            "description": description,
            "status": status,
            "plan_steps": plan_steps or []
        })

    def complete(self):
        """任务完成。"""

        self.status = "completed"

    def fail(self):
        """任务失败。"""

        self.status = "failed"


class ControlState:
    """Agent 控制与保护状态（最小版）。"""

    def __init__(self):

        # 整个运行循环的执行步数计数
        self.execution_count = 0

        # Replan 次数计数
        self.replan_count = 0

        # 上一次失败的签名
        self.last_failure_signature = None

        # 相同失败签名连续出现的次数
        self.repeated_failure_count = 0


class AgentState:
    """统一承载 Agent 运行期的核心状态。

    说明：本步只新增定义，尚未接入 Agent；
    Agent 目前仍使用 Task 与 run() 局部变量，行为完全不变。
    """

    def __init__(self, user_message: str):

        # 任务身份与生命周期
        self.task = Task(user_message)

        # 计划（含每步状态）
        self.plan = []

        # 执行历史（真实结果）
        self.execution_history = []

        # 运行期生命周期状态（pending → running → completed/failed）
        self.status = "pending"

        # 运行期执行流水（Observation/Log）
        self.steps = []

        # 最后一次成功工具调用的结果
        self.last_tool_result = None

        # 控制与保护状态
        self.control = ControlState()