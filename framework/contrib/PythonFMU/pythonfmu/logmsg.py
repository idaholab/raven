
class LogMsg:

    def __init__(self, status: int, category: str, msg: str, debug: bool):
        self.status = status
        self.category = category
        self.msg = msg
        self.debug = debug

    def __str__(self) -> str:
        return "LogMsg(status={}, category={}, msg={}, debug={}".format(self.status, self.category, self.msg, self.debug)



