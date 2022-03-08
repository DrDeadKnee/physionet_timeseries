#!/usr/bin/env python3

import os
import pandas as pd
import yaml
from time import time


class Prepper(object):

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = "config.yml"

        self.config_path = config_path
        self.config = yaml.safe_load(open(config_path, "r"))
        self.started = time()

    def main(self):
        alldirs = [i for i in os.listdir(self.config["raw_data"]) if "training_set" in i]
        outpath = os.path.join(self.config["prepped_data"], "physionet_data.parquet")
        if not os.path.isdir(outpath):
            os.makedirs(outpath)
        big_data = pd.DataFrame()

        count = 0

        for i in alldirs:
            dirname = os.path.join(self.config["raw_data"], i)
            allfiles = os.listdir(dirname)

            for j in range(len(allfiles)):
                raw = pd.read_csv(os.path.join(dirname, allfiles[j]))
                prepped = self.prep_one(raw, j)
                big_data = big_data.append(prepped, ignore_index=True)

                if j % self.config['write_every'] == 0:
                    count += 1
                    big_data.to_parquet(os.path.join(outpath, "checkpooint_{}.parquet".format(j)))
                    big_data = pd.DataFrame()

                if count >= self.config["npackets"]:
                    break

    def get_runtime(self):
        return (time() - self.started) / 60

    def prep_one(self, rawdata, j):
        print(j, self.get_runtime())
        return rawdata


if __name__ == "__main__":
    pp = Prepper()
    pp.main()
