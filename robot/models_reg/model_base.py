class ModelBase:
    """
    the base model class
    """

    def name(self):
        return "ModelBase"

    def initialize(self, opt, device, gpus):
        """
        :param opt: ParameterDict, task settings
        :return: None
        """
        self.opt = opt
        self.devices = device
        self.gpu_ids = gpus
        self.save_dir = opt["path"]["check_point_path"]
        self.record_path = opt["path"]["record_path"]
        self._model = None
        self.iter_count = 0
        self.batch_info = None
        self.caches = {}
        self.lr_scheduler_base_epoch = 0

    def set_input(self, input, device):
        """
        set the input of the method
        :param input:
        :return:
        """
        pass

    def forward(self, input=None):
        pass

    def test(self):
        pass

    def set_train(self):
        """
        set the model in train mode (only for learning methods)
        :return:
        """
        pass

    def set_val(self):
        """
        set the model in validation mode (only for learning methods)
        :return:
        """
        pass

    def set_debug(self):
        """
        set the model in debug (subset of training set) mode (only for learning methods)
        :return:
        """
        pass

    def set_test(self):
        """
        set the model in test mode ( only for learning methods)
        :return:
        """
        pass

    def optimize_parameters(self):
        """
        optimize model parameters
        :return:
        """
        pass

    def get_model(self):
        return self._model

    def get_debug_info(self):
        """ get debug info"""
        return None

    # get image paths
    def get_batch_names(self):
        """get batch name list"""
        pass

    def set_cur_epoch(self, epoch):
        """
        set epoch
        :param epoch:
        :return:
        """
        self.cur_epoch = epoch

    def get_current_errors(self):
        """
        get the current loss
        :return:
        """
        pass

    def get_evaluation(self, input_data):
        """evaluate the performance of the current model"""
        pass

    def update_loss(self, epoch, end_of_epoch):
        pass

    def analyze_res(self, res):
        """
        analyze the results
        :return:
        """
        pass

    def save_res(self, phase):
        pass

    def do_some_clean(self):
        pass
