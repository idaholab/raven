from pythonfmu.fmi2slave import Fmi2Slave, Fmi2Causality, Fmi2Variability, String


class PythonSlaveReadFile(Fmi2Slave):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with (open(f'{self.resources}/hello.txt', 'r')) as file:
            data = file.read()

        self.register_variable(
            String("file_content", getter=lambda: data,
                   causality=Fmi2Causality.output,
                   variability=Fmi2Variability.constant))

    def do_step(self, current_time, step_size):
        return True
