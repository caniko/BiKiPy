import os
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
from pydantic import DirectoryPath, BaseModel


class ExperimentSummary(BaseModel):
    df: pd.DataFrame
    identifier: str
    root_dir_path: Optional[DirectoryPath] = None

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        return self.df

    @property
    def animal_id_indexed_metadata_feature_frame(self):
        return self.df

    @property
    def analysis_path(self):
        self._assert_root_dir()
        os.makedirs(
            (result := self.root_dir_path / f"{self.identifier}_statistics"),
            exist_ok=True,
        )
        return result

    @property
    def figure_path(self):
        os.makedirs((result := self.analysis_path / "figures"), exist_ok=True)
        return result

    def category_vs_features_plot(
        self,
        category: str,
        features: list[str],
    ):
        self._assert_root_dir()

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
        try:
            from statsmodels.multivariate.manova import MANOVA
        except ImportError:
            msg = "Install bikipy[stats] module to perform MANOVA"
            raise ImportError(msg)

        self._assert_root_dir()

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
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
        except ImportError:
            msg = "Install bikipy[stats] module to perform pairwise_tukeyhsd"
            raise ImportError(msg)

        self._assert_root_dir()

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

    def _assert_root_dir(self):
        msg = "root_dir_path needs to be defined for statistical analysis"
        assert self.root_dir_path, msg
