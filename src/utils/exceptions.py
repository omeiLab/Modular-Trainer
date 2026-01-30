# config / validation
class ConfigError(ValueError):
    pass

class HookError(ValueError):
    pass


# data / metric
class MetricError(RuntimeError):
    pass

class ResultBuilderError(TypeError):
    pass


# trainer runtime
class TrainerBaseError(Exception):
    pass

class TrainerError(RuntimeError, TrainerBaseError):
    pass
