#!/usr/bin/env python3

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from time import time
from tqdm import tqdm


class Prepper(object):

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = "config.yml"

        pd.options.mode.chained_assignment = None

        self.config_path = config_path
        self.config = yaml.safe_load(open(config_path, "r"))
        self.started = time()

    def main(self):
        print("\nRunning data prep!")

        alldirs = [i for i in os.listdir(self.config["raw_data"]) if "training_set" in i]
        outpath = os.path.join(self.config["prepped_data"], "physionet_data.parquet")
        if not os.path.isdir(outpath):
            os.makedirs(outpath)
        big_data = pd.DataFrame()

        chunk_count = 0
        user_count = 1

        for i in alldirs:
            dirname = os.path.join(self.config["raw_data"], i)
            allfiles = os.listdir(dirname)

            print("Working on {}".format(i))
            for j in tqdm(range(len(allfiles)), desc="Processing files: "):
                raw = pd.read_csv(os.path.join(dirname, allfiles[j]), delimiter="|")

                try:
                    prepped = self.prep_one(raw, j)
                    prepped["id"] = user_count
                    big_data = big_data.append(prepped, ignore_index=True)
                    user_count += 1
                except (KeyError, ValueError):
                    pass

                if (j % self.config['write_every'] == 0) or (user_count == len(allfiles)):
                    self.cached = big_data.copy()
                    chunk_count += 1
                    table = pa.Table.from_pandas(big_data)
                    pq.write_to_dataset(table, outpath)

                if chunk_count >= self.config["npackets"]:
                    break

            print("Completed {} in {} minutes".format(i, self.get_runtime()))

    def get_runtime(self):
        return (time() - self.started) / 60

    def prep_one(self, rawdata, j):
        if len(self.config["kept_columns"]) > 0:
            rawdata = rawdata[self.config["kept_columns"]]

        for i in self.config["omitted_columns"]:
            del rawdata[i]

        for i in rawdata.columns:
            rawdata[i] = self.impute_values(rawdata[i])

        return rawdata

    def impute_values(self, column):
        imputations = self.config["imputations"]

        if len(column) > sum(column.isna()):
            if imputations["some_nulls"] == "linear_interpolate":
                column = column.interpolate(method="linear", limit_direction="forward")
                column = column.bfill()
            elif imputations["some_nulls"] == "ffill":
                column = column.ffill()
                column = column.bfill()

        elif len(column) == sum(column.isna()):
            column = column.fillna(0)

        return column


if __name__ == "__main__":
    pp = Prepper()
    pp.main()
