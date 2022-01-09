import os
from functools import cached_property

import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from pydantic import BaseModel, DirectoryPath

try:
    from statsmodels.multivariate.manova import MANOVA
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
except ImportError:
    msg = "Install bikipy[stats] module to perform statistical analysis"
    raise ImportError(msg)


rstats = importr("stats")
desctools = importr("DescTools")


class StatisticalAnalysis(BaseModel):
    df: pd.DataFrame
    identifier: str
    root_dir_path: DirectoryPath

    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)

    def __repr__(self):
        return self.df

    def category_vs_features_plot(
        self,
        category: str,
        features: list[str],
    ):
        for parameter in features:
            cat_plot = sb.catplot(
                x=category,
                y=parameter,
                kind="violin",
                inner=None,
                data=self.df,
            )
            sb.swarmplot(
                x=category,
                y=parameter,
                color="k",
                size=3,
                data=self.df,
                ax=cat_plot.ax,
            )
            plt.savefig(
                self.figure_path / f"{category}_vs_{parameter}-{self.identifier}.png"
            )

    def categorical_vs_feature_manova(
        self,
        categories: list[str],
        features: list[str],
    ):
        categories_rhs = " + ".join(map(lambda c: f"C({c})", categories))
        with pd.ExcelWriter(
            self.analysis_path / f"manova_{self.identifier}.xlsx",
            engine_kwargs={
                "strings_to_formulas": False,
                "strings_to_urls": False,
            },
        ) as writer:
            for feature in features:
                analyse = MANOVA.from_formula(
                    f"{categories_rhs} ~ {feature}",
                    self.df,
                )
                analyse.mv_test().summary_frame.to_excel(
                    writer, sheet_name=f"{feature}_{self.identifier}"
                )

    def categorical_vs_feature_pairwise_tukey(
        self,
        categories: list[str],
        features: list[tuple[str, ...]],
    ):
        with pd.ExcelWriter(
            self.analysis_path / f"ad_hoc-tukey_{self.identifier}.xlsx",
            engine_kwargs={
                "strings_to_formulas": False,
                "strings_to_urls": False,
            },
        ) as writer:
            for feature in features:
                tukey_results = []
                for category in categories:
                    tukey = pairwise_tukeyhsd(
                        endog=self.df[feature],  # Data
                        groups=self.df[category],  # Groups
                        alpha=0.05,  # Significance
                    )
                    tukey_results.append(
                        pd.DataFrame(
                            data=tukey._results_table.data[1:],
                            columns=tukey._results_table.data[0],
                        )
                    )
                pd.concat(tukey_results).to_excel(
                    writer, sheet_name=f"{self.identifier}_{features}"
                )

    def categorical_vs_feature_pairwise_bonferroni(
        self,
        categories: list[str],
        features: list[tuple[str, ...]],
    ):
        with pd.ExcelWriter(
            self.analysis_path / f"ad_hoc-bonferroni_{self.identifier}.xlsx",
            engine_kwargs={
                "strings_to_formulas": False,
                "strings_to_urls": False,
            },
        ) as writer:
            for feature in features:
                bonferroni_results = []
                for category in categories:
                    x = rstats.aov(f"{feature} ~ {category}", data=self._r_df)
                    bonferroni = desctools.PostHocTest(
                        x, which=None, method="bonferroni", **{"conf.level": 0.95}
                    )
                    print(bonferroni)

                    bonferroni_results.append(
                        pd.DataFrame(
                            # TODO: Fill me!
                        )
                    )
                pd.concat(bonferroni_results).to_excel(
                    writer, sheet_name=f"{self.identifier}_{features}"
                )

    @cached_property
    def _r_df(self):
        return pandas2ri.py2rpy(self.df)

    @property
    def analysis_path(self):
        os.makedirs(
            (result := self.root_dir_path / f"{self.identifier}_statistics"),
            exist_ok=True,
        )
        return result

    @property
    def figure_path(self):
        os.makedirs((result := self.analysis_path / "figures"), exist_ok=True)
        return result
