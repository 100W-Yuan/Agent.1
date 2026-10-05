import json

from openai import OpenAI

from task import Task, AgentState
from tools import calculator, get_current_time, test_failure
from tool_registry import ToolRegistry


class ExecutionOutcome:
    """run() 内部 Loop 控制结果：Tool Executor 返回给 Loop Controller。

    本步 Tool Executor 实际只产生：
      - CONTINUE               工具执行成功，继续下一轮
      - CONTINUE_AFTER_REPLAN  工具失败且 Replan 成功，继续下一轮
      - FAIL                   致命失败，结束 Agent（已附失败原因）
    FINISH 预留给后续抽离的 Reasoning 分支，本步不产生。
    """

    CONTINUE = "continue"
    CONTINUE_AFTER_REPLAN = "continue_after_replan"
    FINISH = "finish"
    FAIL = "fail"


class Agent:

    def __init__(self, api_key: str, base_url: str, model: str):

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        self.model = model

        self.tool_registry = ToolRegistry()

        self.tool_registry.register(
            "calculator",
            calculator,
            {
                "name": "calculator",
                "description": "执行数学计算",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "要计算的数学表达式"
                        }
                    },
                    "required": ["expression"]
                }
            }
        )

        self.tool_registry.register(
            "get_current_time",
            get_current_time,
            {
                "name": "get_current_time",
                "description": "获取当前时间",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        )

        self.tool_registry.register(
            "test_failure",
            test_failure,
            {
                "name": "test_failure",
                "description": "测试工具失败场景",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        )

    def call_llm(
        self,
        messages,
        tools=None,
        tool_choice="none",
        thinking=True
    ):

        kwargs = {
            "model": self.model,
            "messages": messages
        }

        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        if thinking:
            kwargs["extra_body"] = {
                "thinking": {
                    "type": "enabled"
                }
            }
        else:
            kwargs["extra_body"] = {
                "thinking": {
                    "type": "disabled"
                }
            }

        return self.client.chat.completions.create(
            **kwargs
        )

    def create_plan(
        self,
        user_message: str,
        state: AgentState
    ):

        available_tools = list(
            self.tool_registry.tools.keys()
        )

        prompt = f"""
你是一个 AI Agent 的 Planner。

你的任务是把用户任务拆成严格、可执行的 Plan。

用户任务：
{user_message}

当前系统真实可用工具：
{json.dumps(available_tools, ensure_ascii=False)}

==============================
最重要的规则
==============================

1. 如果某一步需要进行实际计算、获取当前时间或调用系统工具，
   必须定义为 type="tool"。

2. 不允许把“计算数学表达式”定义成 reasoning。

3. 例如：

用户：
计算 25 乘以 4

正确 Plan：

{{
    "steps": [
        {{
            "description": "计算 25 乘以 4",
            "type": "tool",
            "tool_name": "calculator"
        }},
        {{
            "description": "输出计算结果",
            "type": "reasoning",
            "tool_name": null
        }}
    ]
}}

4. calculator 专门用于：
   - 数学计算
   - 四则运算
   - 表达式计算
   - 数字计算

5. get_current_time 专门用于：
   - 获取当前时间
   - 获取当前日期时间

6. test_failure 专门用于：
   - 测试工具失败场景

7. 如果任务明确需要 calculator，
   必须使用 calculator。

8. 如果任务明确需要 get_current_time，
   必须使用 get_current_time。

9. reasoning 只能用于：
   - 提取信息
   - 分析结果
   - 判断
   - 整理信息
   - 最终回答

10. 每一个步骤必须包含：
    - description
    - type
    - tool_name

11. type 只能是：
    - "tool"
    - "reasoning"

12. type="tool" 时：
    tool_name 必须存在于真实工具列表。

13. type="reasoning" 时：
    tool_name 必须为 null。

14. 不允许创造工具。

15. 不要把本来应该由工具完成的事情交给 reasoning。

16. 不要为了减少步骤而省略实际需要的工具调用。

17. 当前任务如果需要工具完成实际操作，
    必须先执行工具，再进行 reasoning。

==============================
返回格式
==============================

只返回合法 JSON：

{{
    "steps": [
        {{
            "description": "步骤描述",
            "type": "tool",
            "tool_name": "工具名称"
        }},
        {{
            "description": "步骤描述",
            "type": "reasoning",
            "tool_name": null
        }}
    ]
}}
"""

        max_plan_attempts = 2
        last_error = None

        for attempt in range(
            1,
            max_plan_attempts + 1
        ):

            print(
                f"\n🧠 Planner 第 {attempt} 次生成 Plan"
            )

            current_prompt = prompt

            if last_error is not None:

                current_prompt += f"""

==============================
上一次 Planner 生成失败
==============================

失败原因：

{last_error}

请重新生成 Plan。

这一次必须严格遵守：

1. 顶层必须是 JSON object。
2. 必须存在 steps 字段。
3. steps 必须是非空列表。
4. 每个 step 必须是 object。
5. 每个 step 必须包含 description、type、tool_name。
6. type 只能是 tool 或 reasoning。
7. tool 步骤必须使用真实存在的工具。
8. reasoning 步骤的 tool_name 必须为 null。

不要解释原因。
不要输出 Markdown。
不要输出 ```json。
只返回合法 JSON。
"""

            response = self.call_llm(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一个严格的 AI Agent Planner。"
                            "必须按照工具使用规则规划任务。"
                            "只返回合法 JSON。"
                        )
                    },
                    {
                        "role": "user",
                        "content": current_prompt
                    }
                ],
                thinking=True
            )

            content = (
                response.choices[0]
                .message
                .content
                or ""
            )

            try:

                data = json.loads(content)

                # ==================================================
                # 第一层验证：顶层必须是 dict
                # ==================================================

                if not isinstance(data, dict):

                    raise ValueError(
                        "Planner 返回的顶层 JSON 不是 object。"
                    )

                # ==================================================
                # 第二层验证：steps 必须存在且为非空 list
                # ==================================================

                steps = data.get("steps")

                if not isinstance(steps, list):

                    raise ValueError(
                        "Planner 返回的 steps 不是列表。"
                    )

                if not steps:

                    raise ValueError(
                        "Planner 没有生成任何步骤。"
                    )

                # ==================================================
                # 第三层验证：完整验证每一个 step
                # ==================================================

                validated_steps = []

                for step in steps:

                    if not isinstance(step, dict):

                        raise ValueError(
                            f"Planner 返回了非法步骤：{step}"
                        )

                    description = step.get("description")
                    step_type = step.get("type")
                    tool_name = step.get("tool_name")

                    if not description:

                        raise ValueError(
                            f"Planner 步骤缺少 description：{step}"
                        )

                    if step_type not in (
                        "tool",
                        "reasoning"
                    ):

                        raise ValueError(
                            "Planner 返回了非法 type："
                            f"{step_type}"
                        )

                    if step_type == "tool":

                        if not tool_name:

                            raise ValueError(
                                "Tool 步骤缺少 tool_name："
                                f"{step}"
                            )

                        if (
                            tool_name
                            not in self.tool_registry.tools
                        ):

                            raise ValueError(
                                "Planner 指定了不存在的工具："
                                f"{tool_name}"
                            )

                    else:

                        # reasoning 必须严格要求 tool_name 为 null。
                        if tool_name is not None:

                            raise ValueError(
                                "Reasoning 步骤的 "
                                "tool_name 必须为 null："
                                f"{step}"
                            )

                    validated_steps.append({
                        "description": description,
                        "type": step_type,
                        "tool_name": tool_name
                    })

            except (
                json.JSONDecodeError,
                ValueError,
                TypeError
            ) as exc:

                last_error = str(exc)

                if isinstance(exc, json.JSONDecodeError):

                    last_error = (
                        "Planner 返回的不是合法 JSON："
                        f"{content}"
                    )

                print(
                    f"❌ Planner 第 {attempt} 次生成失败："
                    f"{last_error}"
                )

                if attempt < max_plan_attempts:

                    print(
                        "🔄 Planner 将自动重试一次"
                    )

                    continue

                raise ValueError(
                    "Planner 连续两次生成有效 Plan 失败："
                    f"{last_error}"
                )

            # ======================================================
            # Plan 完整验证通过后，才写入 state.plan。
            # ======================================================

            for step in validated_steps:

                state.plan.append({
                    "description": step["description"],
                    "type": step["type"],
                    "tool_name": step["tool_name"],
                    "status": "pending"
                })

            print(
                f"✅ Planner 第 {attempt} 次生成成功"
            )

            return

        raise ValueError(
            "Planner 生成 Plan 失败。"
        )

    def replan(
        self,
        state: AgentState,
        failed_tool: str,
        arguments: dict,
        result: dict
    ) -> bool:

        task = state.task

        print(
            "\n========== Agent Replanning =========="
        )

        current_plan = []

        for index, step in enumerate(
            state.plan,
            start=1
        ):

            current_plan.append({
                "index": index,
                "description": step["description"],
                "type": step.get(
                    "type",
                    "reasoning"
                ),
                "tool_name": step.get(
                    "tool_name"
                ),
                "status": step["status"]
            })

        available_tools = list(
            self.tool_registry.tools.keys()
        )

        prompt = f"""
你是一个 AI Agent 的 Replanner。

原始任务：
{task.user_message}

当前 Plan：
{json.dumps(current_plan, ensure_ascii=False)}

当前系统真实可用工具：
{json.dumps(available_tools, ensure_ascii=False)}

刚才调用的工具：
{failed_tool}

工具参数：
{json.dumps(arguments, ensure_ascii=False)}

工具返回结果：
{json.dumps(result, ensure_ascii=False)}

工具调用失败了。

请重新设计后续执行计划。

要求：

1. 必须考虑工具失败原因。
2. 不要假设失败工具已经成功。
3. 可以跳过无法执行的步骤。
4. 可以使用其他工具。
5. 可以增加 reasoning 步骤。
6. 每个步骤必须包含：
   - description
   - type
   - tool_name
7. type 只能是：
   - "tool"
   - "reasoning"
8. 如果 type == "tool"：
   - tool_name 必须填写。
   - tool_name 必须来自当前系统真实可用工具列表。
   - 严禁创造不存在的工具。
9. 如果 type == "reasoning"：
   - tool_name 必须为 null。
10. 新 Plan 必须能够继续推进用户原始任务。
11. 不要重复执行已经成功完成的任务步骤。
12. 只返回 JSON。

格式：

{{
    "steps": [
        {{
            "description": "步骤描述",
            "type": "tool",
            "tool_name": "真实存在的工具名称"
        }},
        {{
            "description": "步骤描述",
            "type": "reasoning",
            "tool_name": null
        }}
    ]
}}
"""

        response = self.call_llm(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个严格的 AI Agent Replanner，"
                        "只返回合法 JSON。"
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            thinking=True
        )

        content = (
            response.choices[0]
            .message
            .content
            or ""
        )

        try:

            data = json.loads(content)

            # 顶层必须是 object
            if not isinstance(data, dict):

                raise ValueError(
                    "Replanning 返回的顶层 JSON 不是 object。"
                )

            steps = data.get("steps")

            if not isinstance(steps, list):

                raise ValueError(
                    "Replanning 返回的 steps 不是列表。"
                )

            if not steps:

                raise ValueError(
                    "Replanning 没有生成任何步骤。"
                )

            new_plan = []

            for step in steps:

                if not isinstance(step, dict):

                    raise ValueError(
                        f"Replanning 存在非法步骤：{step}"
                    )

                description = step.get("description")
                step_type = step.get("type")
                tool_name = step.get("tool_name")

                if not description:

                    raise ValueError(
                        "Replanning 步骤缺少 description："
                        f"{step}"
                    )

                if step_type not in (
                    "tool",
                    "reasoning"
                ):

                    raise ValueError(
                        "Replanning 存在非法 type："
                        f"{step_type}"
                    )

                if step_type == "tool":

                    if not tool_name:

                        raise ValueError(
                            "Tool 步骤缺少 tool_name："
                            f"{step}"
                        )

                    if (
                        tool_name
                        not in self.tool_registry.tools
                    ):

                        raise ValueError(
                            "Replanning 指定了不存在的工具："
                            f"{tool_name}"
                        )

                else:

                    if tool_name is not None:

                        raise ValueError(
                            "Reasoning 步骤的 "
                            "tool_name 必须为 null："
                            f"{step}"
                        )

                new_plan.append({
                    "description": description,
                    "type": step_type,
                    "tool_name": tool_name,
                    "status": "pending"
                })

        except (
            json.JSONDecodeError,
            ValueError,
            TypeError
        ) as exc:

            print(
                f"❌ Replanning Plan 校验失败：{exc}"
            )

            return False

        state.plan = new_plan

        print(
            "🔄 Replanning 结果："
            + json.dumps(
                data,
                ensure_ascii=False
            )
        )

        print("📋 新 Plan：")

        for index, step in enumerate(
            state.plan,
            start=1
        ):

            print(
                f"  {index}. "
                f"{step['description']} "
                f"[{step['type']}] "
                f"[tool={step.get('tool_name')}] "
                f"[{step['status']}]"
            )

        print(
            "✅ Replanning 完成，Agent 继续执行"
        )

        return True

    def get_current_plan_step(
        self,
        state: AgentState
    ):

        for index, step in enumerate(
            state.plan,
            start=1
        ):

            if step["status"] != "completed":

                return {
                    "index": index,
                    "description": step["description"],
                    "type": step.get(
                        "type",
                        "reasoning"
                    ),
                    "tool_name": step.get(
                        "tool_name"
                    ),
                    "status": step["status"]
                }

        return None

    def build_plan_execution_context(
        self,
        state: AgentState,
        execution_outputs=None
    ) -> str:

        current_step = (
            self.get_current_plan_step(state)
        )

        lines = []

        lines.append(
            "当前任务执行计划："
        )

        for index, step in enumerate(
            state.plan,
            start=1
        ):

            lines.append(
                f"{index}. "
                f"{step['description']} "
                f"[{step.get('type', 'reasoning')}] "
                f"[tool={step.get('tool_name')}] "
                f"[{step['status']}]"
            )

        if execution_outputs:

            lines.append("")
            lines.append(
                "历史执行记录及真实结果："
            )

            for output in execution_outputs:

                lines.append(
                    f"第 {output['step']} 步："
                    f"{output['description']}"
                )

                lines.append(
                    f"类型：{output['type']}"
                )

                lines.append(
                    f"执行状态：{output.get('status', 'unknown')}"
                )

                lines.append(
                    f"结果：{output['output']}"
                )

        if current_step:

            lines.append("")

            lines.append(
                f"当前必须优先处理："
                f"第 {current_step['index']} 步"
            )

            lines.append(
                f"当前步骤："
                f"{current_step['description']}"
            )

            lines.append(
                f"当前步骤类型："
                f"{current_step['type']}"
            )

            lines.append(
                f"当前指定工具："
                f"{current_step.get('tool_name')}"
            )

            lines.append("")

            lines.append(
                "不要跳过当前步骤直接执行后面的步骤。"
            )

        return "\n".join(lines)

    def judge_reasoning_step(
        self,
        state: AgentState,
        current_step: dict,
        content: str
    ) -> bool:

        task = state.task

        prompt = f"""
你是 AI Agent 的当前步骤完成判断器。

原始任务：
{task.user_message}

当前 Plan：
{json.dumps(state.plan, ensure_ascii=False)}

当前正在执行：

第 {current_step['index']} 步
描述：
{current_step['description']}

类型：
reasoning

Agent 当前输出：

{content}

请判断：

当前 Agent 输出是否已经真正完成
“第 {current_step['index']} 步”。

严格要求：

1. 只能判断当前步骤。
2. 不允许判断后续步骤。
3. 如果当前输出已经完成当前步骤，
   返回 true。
4. 如果当前输出没有完成当前步骤，
   返回 false。

只返回 JSON：

{{
    "completed": true
}}

或者：

{{
    "completed": false
}}
"""

        response = self.call_llm(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是严格的 Agent 步骤完成判断器。"
                        "只返回合法 JSON。"
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            thinking=True
        )

        content_result = (
            response.choices[0]
            .message
            .content
            or ""
        )

        try:

            data = json.loads(content_result)

        except json.JSONDecodeError:

            print(
                f"❌ Reasoning 判断 JSON 解析失败："
                f"{content_result}"
            )

            return False

        if not isinstance(data, dict):

            print(
                "❌ Reasoning 判断结果顶层不是 object"
            )

            return False

        completed = data.get(
            "completed",
            False
        )

        print(
            "📋 Reasoning 完成判断："
            + json.dumps(
                data,
                ensure_ascii=False
            )
        )

        return completed is True

    def finish_task(
        self,
        state: AgentState
    ):

        state.status = "completed"

        print(
            "✅ Agent 判断任务已经完成"
        )

        print(
            f"📋 Task 状态："
            f"{state.status}"
        )

        print(
            f"📋 Task Plan："
            f"{state.plan}"
        )

        print(
            "📋 Execution："
        )

        for execution in state.steps:

            print(
                f"  Plan "
                f"{execution['plan_steps']} "
                f"→ "
                f"{execution['description']} "
                f"→ "
                f"{execution['status']}"
            )

    def execute_tool_step(
        self,
        state: AgentState,
        current_step: dict,
        tool_messages: list,
        max_replan_count: int,
        max_repeated_failure_count: int
    ):

        required_tool = (
            current_step.get(
                "tool_name"
            )
        )

        if not required_tool:

            state.status = "failed"

            return ExecutionOutcome.FAIL, (
                "当前 Plan 缺少指定工具，"
                "任务无法继续。"
            )

        if (
            required_tool
            not in self.tool_registry.tools
        ):

            state.status = "failed"

            return ExecutionOutcome.FAIL, (
                f"当前 Plan 指定的工具 "
                f"{required_tool} 不存在。"
            )

        plan_context = (
            self.build_plan_execution_context(
                state,
                state.execution_history
            )
        )

        current_tool_messages = (
            list(tool_messages)
        )

        current_tool_messages.append({
            "role": "system",
            "content": plan_context
        })

        available_tools = []

        for tool in (
            self.tool_registry
            .get_openai_tools()
        ):

            if (
                tool["function"]["name"]
                == required_tool
            ):

                available_tools.append(
                    tool
                )

                break

        if not available_tools:

            state.status = "failed"

            return ExecutionOutcome.FAIL, (
                f"无法找到工具定义："
                f"{required_tool}"
            )

        tool_choice = {
            "type": "function",
            "function": {
                "name": required_tool
            }
        }

        print(
            f"🔒 工具权限锁定："
            f"{required_tool}"
        )

        response = self.call_llm(
            messages=current_tool_messages,
            tools=available_tools,
            tool_choice=tool_choice,
            thinking=False
        )

        assistant_message = (
            response.choices[0].message
        )

        if not assistant_message.tool_calls:

            print(
                "❌ Tool 步骤没有产生工具调用"
            )

            state.status = "failed"

            return ExecutionOutcome.FAIL, (
                "当前 Plan 要求调用工具，"
                "但模型没有产生工具调用。"
            )

        did_replan = False
        for tool_call in (
            assistant_message.tool_calls
        ):

            tool_name = (
                tool_call.function.name
            )

            if (
                tool_name
                != current_step.get(
                    "tool_name"
                )
            ):

                print(
                    "❌ 工具调用违反当前 Plan："
                    f"{tool_name}"
                )

                state.status = "failed"

                return ExecutionOutcome.FAIL, (
                    "Agent 尝试调用未被当前 "
                    "Plan 允许的工具。"
                )

            try:

                arguments = json.loads(
                    tool_call.function.arguments
                )

            except json.JSONDecodeError:

                print(
                    "❌ 工具参数 JSON 解析失败"
                )

                state.status = "failed"

                return ExecutionOutcome.FAIL, (
                    "工具参数不是合法 JSON。"
                )

            print(
                f"🔧 调用工具："
                f"{tool_name}"
            )

            print(
                f"📥 工具参数："
                f"{arguments}"
            )

            result = (
                self.tool_registry.execute(
                    tool_name,
                    arguments
                )
            )

            print(
                f"📤 工具结果："
                f"{result}"
            )

            if not result.get("success"):

                print(
                    "❌ 工具执行失败"
                )

                # ==================================================
                # 重复失败检测
                # ==================================================

                failure_signature = (
                    tool_name,
                    json.dumps(
                        arguments,
                        ensure_ascii=False,
                        sort_keys=True
                    ),
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        sort_keys=True
                    )
                )

                if (
                    failure_signature
                    == state.control.last_failure_signature
                ):

                    state.control.repeated_failure_count += 1

                else:

                    state.control.last_failure_signature = (
                        failure_signature
                    )

                    state.control.repeated_failure_count = 1

                print(
                    "🛡️ 重复失败计数："
                    f"{state.control.repeated_failure_count}/"
                    f"{max_repeated_failure_count}"
                )

                state.steps.append({
                    "description": f"调用工具：{tool_name}",
                    "status": "failed",
                    "plan_steps": [current_step["index"]]
                })

                current_index = (
                    current_step["index"]
                )

                state.execution_history.append({
                    "step": current_index,
                    "description": current_step["description"],
                    "type": "tool",
                    "status": "failed",
                    "output": result
                })

                if (
                    state.control.repeated_failure_count
                    >= max_repeated_failure_count
                ):

                    print(
                        "🛑 相同工具、相同参数、"
                        "相同失败结果重复出现，"
                        "触发重复失败熔断"
                    )

                    state.status = "failed"

                    return ExecutionOutcome.FAIL, (
                        "同一个工具调用重复失败，"
                        "Agent 已触发重复失败熔断，"
                        "任务停止。"
                        "失败原因："
                        + json.dumps(
                            result,
                            ensure_ascii=False
                        )
                    )

                print(
                    "🧠 Agent 执行上下文更新："
                )

                print(
                    json.dumps(
                        state.execution_history,
                        ensure_ascii=False,
                        indent=2
                    )
                )

                # ==================================================
                # 独立 Replan 上限
                # ==================================================

                if (
                    state.control.replan_count
                    >= max_replan_count
                ):

                    print(
                        "🛑 Replan 次数达到独立上限："
                        f"{state.control.replan_count}/"
                        f"{max_replan_count}"
                    )

                    state.status = "failed"

                    return ExecutionOutcome.FAIL, (
                        "Replanning 次数超过限制，"
                        "任务未完成。"
                        "失败原因："
                        + json.dumps(
                            result,
                            ensure_ascii=False
                        )
                    )

                state.control.replan_count += 1

                print(
                    "🔄 当前 Replan 次数："
                    f"{state.control.replan_count}/"
                    f"{max_replan_count}"
                )

                replanned = self.replan(
                    state,
                    tool_name,
                    arguments,
                    result
                )

                if not replanned:

                    state.status = "failed"

                    return ExecutionOutcome.FAIL, (
                        "工具执行失败，"
                        "Replanning 也失败。"
                        "失败原因："
                        + json.dumps(
                            result,
                            ensure_ascii=False
                        )
                    )

                did_replan = True
                continue

            state.last_tool_result = result.get(
                "result"
            )

            current_index = (
                current_step["index"]
            )

            state.execution_history.append({
                "step": current_index,
                "description": current_step["description"],
                "type": "tool",
                "status": "completed",
                "output": state.last_tool_result
            })

            print(
                "🧠 Agent 执行上下文更新："
            )

            print(
                json.dumps(
                    state.execution_history,
                    ensure_ascii=False,
                    indent=2
                )
            )

            state.plan[
                current_index - 1
            ]["status"] = "completed"

            print(
                f"📋 Plan 更新："
                f"第 {current_index} 步完成"
            )

            print(
                "📋 当前 Plan："
            )

            for index, step in enumerate(
                state.plan,
                start=1
            ):

                print(
                    f"  {index}. "
                    f"{step['description']} "
                    f"[{step['type']}] "
                    f"[tool={step.get('tool_name')}] "
                    f"[{step['status']}]"
                )

            state.steps.append({
                "description": f"调用工具：{tool_name}",
                "status": "completed",
                "plan_steps": [current_index]
            })

        if did_replan:
            return ExecutionOutcome.CONTINUE_AFTER_REPLAN, None
        return ExecutionOutcome.CONTINUE, None

    def execute_reasoning_step(
        self,
        state: AgentState,
        current_step: dict
    ):

        task = state.task

        print(
            "🧠 当前为 reasoning 步骤："
            "禁止工具调用"
        )

        previous_outputs = json.dumps(
            state.execution_history,
            ensure_ascii=False,
            indent=2
        )

        reasoning_prompt = f"""
你现在是 AI Agent 的最终回答阶段。

用户原始任务：
{task.user_message}

当前完整 Plan：
{json.dumps(state.plan, ensure_ascii=False)}

历史执行记录及真实结果：

{previous_outputs}

当前步骤：

第 {current_step['index']} 步
描述：
{current_step['description']}

请根据已经得到的真实执行结果，
完成当前 reasoning 步骤。

要求：

1. 只能基于已经得到的信息回答。
2. 不要再次调用工具。
3. 不要假装执行了其他工具。
4. 必须使用前面步骤已经得到的真实结果。
5. 当前步骤只负责完成当前 Plan 步骤。
6. 不要擅自修改前面工具得到的结果。
7. 如果历史记录中存在工具失败结果，
   必须承认该工具确实失败。
8. 不要把失败工具描述成成功执行。
"""

        reasoning_messages = [
            {
                "role": "system",
                "content": (
                    "你是 AI Agent 的最终回答模块。"
                    "当前阶段禁止工具调用。"
                )
            },
            {
                "role": "user",
                "content": reasoning_prompt
            }
        ]

        response = self.call_llm(
            messages=reasoning_messages,
            tools=[],
            tool_choice="none",
            thinking=True
        )

        assistant_message = (
            response.choices[0].message
        )

        if assistant_message.tool_calls:

            print(
                "❌ Reasoning 步骤出现工具调用"
            )

            state.status = "failed"

            return ExecutionOutcome.FAIL, (
                "Agent 在 reasoning 步骤尝试调用工具，"
                "违反当前 Plan。"
            )

        final_content = (
            assistant_message.content
            or ""
        )

        print(
            f"💬 Agent Reasoning 输出："
            f"{final_content}"
        )

        completed = (
            self.judge_reasoning_step(
                state,
                current_step,
                final_content
            )
        )

        if not completed:

            print(
                "⚠️ 当前 reasoning 步骤尚未完成，"
                "重新进行 reasoning。"
            )

            return ExecutionOutcome.CONTINUE, None

        current_index = (
            current_step["index"]
        )

        state.execution_history.append({
            "step": current_index,
            "description": current_step["description"],
            "type": "reasoning",
            "status": "completed",
            "output": final_content
        })

        print(
            "🧠 Agent 执行上下文更新："
        )

        print(
            json.dumps(
                state.execution_history,
                ensure_ascii=False,
                indent=2
            )
        )

        state.plan[
            current_index - 1
        ]["status"] = "completed"

        print(
            f"📋 Plan 更新："
            f"第 {current_index} 步完成"
        )

        print(
            "📋 当前 Plan："
        )

        for index, step in enumerate(
            state.plan,
            start=1
        ):

            print(
                f"  {index}. "
                f"{step['description']} "
                f"[{step['type']}] "
                f"[tool={step.get('tool_name')}] "
                f"[{step['status']}]"
            )

        state.steps.append({
            "description": "Agent 完成推理步骤",
            "status": "completed",
            "plan_steps": [current_index]
        })

        all_completed = all(
            step["status"] == "completed"
            for step in state.plan
        )

        if all_completed:

            state.steps.append({
                "description": "Agent 完成最终回答",
                "status": "completed",
                "plan_steps": [current_index]
            })

            self.finish_task(state)

            return ExecutionOutcome.FINISH, final_content

        return ExecutionOutcome.CONTINUE, None

    def run(
        self,
        user_message: str
    ):

        print(
            "\n========== Agent 开始 =========="
        )

        print(
            f"用户任务：{user_message}"
        )

        state = AgentState(user_message)

        state.status = "running"

        self.create_plan(
            user_message,
            state
        )

        print("📋 执行计划：")

        for index, step in enumerate(
            state.plan,
            start=1
        ):

            print(
                f"  {index}. "
                f"{step['description']} "
                f"[{step.get('type', 'reasoning')}] "
                f"[tool={step.get('tool_name')}]"
            )

        # ==========================================================
        # 两层执行保护
        #
        # max_execution_count：
        # 防止整个 Agent 无限运行。
        #
        # max_replan_count：
        # 专门限制 Replanning 次数。
        # 两者独立。
        # ==========================================================

        max_execution_count = 20

        max_replan_count = 5

        # ==========================================================
        # 重复失败保护
        #
        # key：
        # tool_name + arguments + failure result
        #
        # 同一个失败组合连续出现 2 次：
        # 直接熔断。
        # ==========================================================

        max_repeated_failure_count = 2

        tool_messages = [
            {
                "role": "system",
                "content": """
你是一个能够调用工具的 AI Agent。

你必须严格按照当前 Plan 执行任务。

如果当前步骤是 tool：

必须调用当前 Plan 指定的工具。

不要自己完成工具应该完成的任务。
"""
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        while state.control.execution_count < max_execution_count:

            state.control.execution_count += 1

            print(
                f"\n🧠 Agent 第 "
                f"{state.control.execution_count} 步"
            )

            current_step = (
                self.get_current_plan_step(state)
            )

            if current_step is None:

                self.finish_task(state)

                if state.last_tool_result is not None:

                    return str(state.last_tool_result)

                return ""

            print(
                f"🎯 当前 Plan：第 "
                f"{current_step['index']} 步"
            )

            print(
                f"🎯 当前步骤："
                f"{current_step['description']}"
            )

            print(
                f"🎯 当前类型："
                f"{current_step['type']}"
            )

            print(
                f"🎯 指定工具："
                f"{current_step.get('tool_name')}"
            )

            if current_step["type"] == "tool":

                outcome, reply = self.execute_tool_step(
                    state,
                    current_step,
                    tool_messages,
                    max_replan_count,
                    max_repeated_failure_count
                )

                if outcome == ExecutionOutcome.FAIL:

                    return reply

                continue

            outcome, reply = self.execute_reasoning_step(
                state,
                current_step
            )

            if outcome == ExecutionOutcome.CONTINUE:

                continue

            return reply

        state.status = "failed"

        print(
            "❌ Agent 达到最大执行次数，"
            "任务失败"
        )

        print(
            f"📋 Task 状态："
            f"{state.status}"
        )

        return (
            "Agent 执行次数超过限制，"
            "任务未完成。"
        )