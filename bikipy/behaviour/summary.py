import os
from functools import cached_property
from typing import Iterable, Union

import numpy as np
import pandas as pd
from pydantic import DirectoryPath, root_validator

from bikipy.core.base_class import BikipyBase

try:
    from statsmodels.multivariate.manova import MANOVA
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
except ImportError:
    msg = "Install bikipy[stats] module to perform statistical analysis"
    raise ImportError(msg)


class StatisticalAnalysis(BikipyBase):
    df: pd.DataFrame
    category_columns: Iterable[str, ...]
    identifier: str
    root_dir_path: DirectoryPath

    @classmethod
    def merge_analysis_data_with_metadata(cls, analysis_df: pd.DataFrame, df: pd.DataFrame, animal_id_column_name: str = "Animal ID", **kwargs):
        if animal_id_column_name in df.columns:
            df = df.set_index(animal_id_column_name).sort_index()
        elif animal_id_column_name != df.index.name:
            msg = f"The metadata does not define the given animal_id_column_name, {animal_id_column_name}"
            raise KeyError(msg)
        df.index.name = "Animal ID"
        return cls(df=pd.merge(analysis_df, df, on="Animal ID"), **kwargs)

    def __repr__(self):
        return self.df

    @root_validator
    def ensure_metadata_df_is_clean(cls, values):
        df = values["df"]
        values["df"] = (
            df.drop_duplicates(values["animal_id_column_name"])
            .set_index(values["animal_id_column_name"])
            .applymap(lambda x: x.strip() if isinstance(x, str) else x)
        )
        return values

    @cached_property
    def unique_category_values(self):
        result = {}
        for column in self.category_columns:
            assert column in self.df.columns
            result[column] = np.unique(self.df[column])
        return result

    @cached_property
    def categorized_dataframes(self) -> pd.DataFrame:
        dataframes = []
        for column in self.category_columns:
            median_series = []
            for unique_category in self.unique_category_values[column]:
                boolean_index = self.df[column] == unique_category
                assert np.any(boolean_index)
                median_series.append(self.df.iloc[boolean_index, :].median())
            dataframes.append(
                pd.DataFrame(
                    median_series,
                    index=pd.MultiIndex.from_product(
                        [[column], self.unique_category_values[column]]
                    ),
                )
            )
        return pd.concat(dataframes, axis=0)

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
                    bonferroni_results.append(
                        pd.DataFrame(
                            # TODO: Fill me!
                        )
                    )
                pd.concat(bonferroni_results).to_excel(
                    writer, sheet_name=f"{self.identifier}_{features}"
                )

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
