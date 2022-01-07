import os
import re
import sys

def analysis_performance(path):
    """ Extract throughput from TF keras log
    """

    dir = os.listdir(path)
    logs = [fname for fname in dir if str(fname).__contains__(".log")]
    logs = sorted(logs, key=lambda name: int(re.findall(r"\d+", name)[-1]))
    for log in logs:
        log_name = os.path.join(path, log)
        print("Extract log: {}".format(log_name))
        with open(log_name, "r") as f:
            line = f.readline()
            while line:
                if line.__contains__("====") and line.__contains__("step"):
                    num_batches = int(re.findall(r"\d+/\d+", line)[0].split("/")[0])
                    batch_size = int(re.findall(r"\d+", log_name)[-1])
                    epoch_time = int(re.findall(r"\d+s ", line)[0][:-2])
                    throughput = 1 / (epoch_time / num_batches) * batch_size
                    print("Batch num: {}, batch size: {}, total time: {}s, throughput: {} img/s\n"
                                                .format(num_batches, batch_size, epoch_time, throughput))
                    break
                line = f.readline()
                    
if __name__ == "__main__":
    log = sys.argv[1]
    analysis_performance(log)
