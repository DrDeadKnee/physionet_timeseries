#!/usr/bin/env python3

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shutil
import sys
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
        self.means = pd.read_csv("summary_data/mean_values.csv")

    def main(self):
        print("\nRunning data prep!")

        alldirs = [i for i in os.listdir(self.config["raw_data"]) if "training_set" in i]
        outpath = os.path.join(self.config["prepped_data"], "physionet_data.parquet")

        if os.path.exists(outpath):
            self.clear_previous(outpath)

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
                    if len(prepped.index) > self.config["data_length"]["min_length"]:
                        big_data = big_data.append(prepped, ignore_index=True)
                    user_count += 1
                    print(big_data)
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

    def clear_previous(self, outpath):
        print("Need to remove existing data at {}".format(outpath))
        if yesno("Is it ok to proceed?"):
            shutil.rmtree(outpath)
        else:
            print("Exiting...")
            sys.exit()

    def get_runtime(self):
        return (time() - self.started) / 60

    def prep_one(self, rawdata, j):
        if self.config["remove_nonkept"]:
            rawdata = rawdata[self.config["kept_columns"]]

        for i in self.config["omitted_columns"]:
            del rawdata[i]

        for i in rawdata.columns:
            rawdata[i] = self.impute_values(rawdata[i], i)

        rawdata["SepsisEver"] = min(1, max(0, -0.9 + sum(rawdata["SepsisLabel"])))

        return rawdata

    def impute_values(self, column, columnname):
        imputations = self.config["imputations"]

        if len(column) > sum(column.isna()):
            if imputations["some_nulls"] == "linear_interpolate":
                column = column.interpolate(method="linear", limit_direction="forward")
                column = column.ffill()
                column = column.bfill()
            elif imputations["some_nulls"] == "ffill":
                column = column.ffill()
                column = column.bfill()
            else:
                raise BadArgumentError("imputations: some_nulls must be in 'ffill' or 'linear_interpolate'")

        elif len(column) == sum(column.isna()):
            if (imputations["all_nulls"] == "global_mean") and (columnname in self.means.columns):
                column = column.fillna(self.means[columnname].iloc[0])
            else:
                column = column.fillna(0)

        return column


def yesno(query):
    response = input(query + "(y/n)\n>")
    if response == "y":
        return True
    elif response == "n":
        return False
    else:
        print("Response must be 'y' or 'n'")
        return yesno(query)


class BadArgumentError(Exception):
    pass


if __name__ == "__main__":
    pp = Prepper()
    pp.main()
