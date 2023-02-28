class Constants:
    """
    Class for all constant parameters.
    """

    def __init__(self, subpath):
        self.subpath = subpath
        self.PATH = "./Experiment/" + str(self.subpath)

        self.DATASETS = [
            "artificial_dataset",
        ]

        self.NAME_METHODS = [
            "phcp",
            
        ]

        # parameter for phcp
        # Random forest iterations and number of CV folds
        self.iterations_rf = 100
        # cv folds
        self.cv = 10
        # Boolean variables for optimization
        self.to_optimize = True
        # HO hyperparameter, steps of iteration
        self.ho_max_step = 2
        # enable incremental result output
        self.stats_inc_output = True
