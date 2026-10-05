class ToolRegistry:

    def __init__(self):
        self.tools = {}

    def register(self, name, function, definition):
        self.tools[name] = {
            "function": function,
            "definition": definition
        }

    def get_definitions(self):
        return [
            tool["definition"]
            for tool in self.tools.values()
        ]

    def get_openai_tools(self):
        """
        返回 OpenAI / DeepSeek Chat Completions API
        所需要的 tools 格式。
        """

        return [
            {
                "type": "function",
                "function": tool["definition"]
            }
            for tool in self.tools.values()
        ]

    def execute(self, name, arguments):

        # 工具不存在
        if name not in self.tools:
            return {
                "success": False,
                "error": f"未知工具：{name}"
            }

        function = self.tools[name]["function"]

        try:
            # 执行工具
            result = function(**arguments)

            return {
                "success": True,
                "result": result
            }

        except Exception as exc:
            # 工具执行失败
            return {
                "success": False,
                "error": str(exc)
            }