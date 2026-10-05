from datetime import datetime


def calculator(expression: str) -> str:
    """
    计算数学表达式。
    支持常见的 ^ 乘方写法，并转换为 Python 的 **。
    """

    expression = expression.replace("^", "**")

    result = eval(expression, {"__builtins__": {}}, {})

    return str(result)


def get_current_time() -> str:
    """
    获取当前时间。
    """

    now = datetime.now()

    return now.strftime("%Y-%m-%d %H:%M:%S")


def test_failure():
    """
    测试工具失败处理。
    """
    raise RuntimeError("这是一次故意制造的工具错误")