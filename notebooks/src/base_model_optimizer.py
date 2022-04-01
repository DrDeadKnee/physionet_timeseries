from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_validate
import pandas as pd


class BoostModel(object):

    def __init__(self, X, y, group="none", nparts=10):
        self.Xdata = X
        self.ydata = y
        self.nparts = nparts
        self.results = pd.DataFrame()
        self.dgroup = group

    def optimize(self):
        depths = [2, 10, 20]
        min_leafs = [2, 32, 128]

        for i in depths:
            for j in min_leafs:
                model = RFC(max_depth=i, min_samples_leaf=j, class_weight="balanced")
                cv_results = cross_validate(model, self.Xdata, self.ydata, cv=self.nparts, scoring=("f1", "accuracy"))
                self.results = self.results.append(
                    {
                        "depth": i,
                        "min_leafs": j,
                        "data_group": self.dgroup,
                        "test_acc": cv_results["test_accuracy"].mean(),
                        "test_acc_std": cv_results["test_accuracy"].std(),
                        "test_f1": cv_results["test_accuracy"].mean(),
                        "test_f1_std": cv_results["test_accuracy"].std(),
                    },
                    ignore_index=True
                )
                print(
                    "Random Forest with depth {} and min_leafs {} got accuracy = {} +/- {} with f1 = {} on {}.".format(
                        i, j,
                        cv_results["test_accuracy"].mean(), cv_results["test_accuracy"].std(),
                        cv_results["test_f1"],
                        self.dgroup
                    )
                )

    def get_optimized(self, param="f1"):
        try:
            assert(len(self.results.index) > 0)
        except AssertionError:
            raise ValueError("BoostModel.get_optimized requires that optimize be run first")

        optparams = self.results.sort_values(by=f"test_{param}").iloc[0]
        optimized = RFC(
            max_depth=int(optparams["depth"]),
            min_samples_leaf=int(optparams["min_leafs"]),
            class_weight="balanced"
        )
        optimized.fit(self.Xdata, self.ydata)

        return optimized
