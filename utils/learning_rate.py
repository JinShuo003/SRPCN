class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):
    schedule_specs = specs["LearningRateSchedule"]

    if schedule_specs["Type"] == "Step":
        schedule = StepLearningRateSchedule(schedule_specs["Initial"], schedule_specs["Interval"],
                                            schedule_specs["Factor"])
    elif schedule_specs["Type"] == "Warmup":
        schedule = WarmupLearningRateSchedule(schedule_specs["Initial"], schedule_specs["Final"],
                                              schedule_specs["Length"])
    elif schedule_specs["Type"] == "Constant":
        schedule = ConstantLearningRateSchedule(schedule_specs["Value"])
    else:
        raise Exception('no known learning rate schedule of type "{}"'.format(schedule_specs["Type"]))

    return schedule