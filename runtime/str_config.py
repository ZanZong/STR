import shutil
import os

# Default config is in the user directory
CONFIG_DIR = os.path.join(os.environ['HOME'], ".str")

class ConfigHandler:
    def __init__(self, model_name, batch_size) -> None:
        self.model_name = model_name
        self.bs = batch_size
        
        dir = os.listdir(CONFIG_DIR)
        dir_name = [name for name in dir if str(name).__contains__(model_name)]

        if len(dir_name) > 0:
            bs_exist = [int(name.split("_")[-1]) for name in dir_name] # extract batch size from dir name
            if batch_size in bs_exist:
                base_dir = model_name + "_" + str(batch_size)
            else:
                # use the config with largest batch size
                base_dir = model_name + "_" + str(max(bs_exist))
                print("Cannot find config for batch_size={}, use {} instead".format(batch_size, max(bs_exist)))
            self.model_base_dir = os.path.join(CONFIG_DIR, base_dir)
        else:
            print("Cannot find config dir for {} in {}".format(model_name, CONFIG_DIR))
        
    def set_strategy(self, name: str):
        """ Set the current tensorflow strategy w.r.t the `name`
        """
        if not hasattr(self, "model_base_dir") or \
            name not in ["str", "dynprog", "checkmate", "capuchin", "chen-heurist", "str-app"]:
            shutil.copy(os.path.join(CONFIG_DIR, "origin.conf"), \
                                    os.path.join(CONFIG_DIR, "strategy.conf"))
            print("Using original tensorflow without memory optimization")
        else:
            assert name in ["str", "dynprog", "checkmate", "capuchin", "chen-heurist", "str-app"], "Strategy not supported!"
            shutil.copy(os.path.join(self.model_base_dir, name + ".conf"), \
                                    os.path.join(CONFIG_DIR, "strategy.conf"))

if __name__ == "__main__":
    chandle = ConfigHandler("VGG16", 240)
    chandle.set_strategy("str")