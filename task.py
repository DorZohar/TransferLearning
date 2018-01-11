
TASK_TYPE__SEQUENCE_TAGGING = 0
TASK_TYPE__SENTENCE_CLASSIFICATION = 1
TASK_TYPE__TWO_SENTENCES_CLASSIFICATION = 2


class Task:

    def __init__(self, conf):
        self.task_id = conf['cl_task_current']
        self.name = conf['cl_task__names'][self.task_id]
        self.num_classes = conf['cl_task__classes'][self.task_id]
        self.type = conf['cl_task__type'][self.task_id]

    def get_corpus_path(self):
        pass

    def extract_features(self):
        pass

    def get_model_path(self):
        pass