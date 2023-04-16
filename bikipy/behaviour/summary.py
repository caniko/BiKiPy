import os
from functools import cached_property

import numpy as np
import pandas as pd
from pydantic import DirectoryPath, validator

from bikipy.core.base_class import BikipyModel

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
except ImportError:
    msg = "Install bikipy[stats] module to perform statistical analysis"
    raise ImportError(msg)


class StatisticalAnalysis(BikipyModel):
    analysis_df: pd.DataFrame
    metadata_df: pd.DataFrame
    category_columns: tuple[str, ...]
    feature_columns: tuple
    identifier: str
    project_directory: DirectoryPath

    @validator("metadata_df")
    def ensure_metadata_df_is_clean(cls, value):
        return value.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    def __repr__(self):
        return self.analysis_df

    def categorical_to_feature_pairwise_tukey(self):
        result = []
        for category in self.category_columns:
            for feature_column in self.feature_columns:
                tukey = pairwise_tukeyhsd(
                    endog=self.analysis_df.loc[:, feature_column],  # Data
                    groups=self.metadata_df.loc[:, category],  # Groups
                    alpha=0.05,  # Significance
                )
                tukey_df = pd.DataFrame(
                    data=[row[2:] for row in tukey._results_table.data[1:]],
                    columns=tukey._results_table.data[0][2:],
                    index=["{}_{}".format(*row[:2]) for row in tukey._results_table.data[1:]],
                )
                tukey_df.index = pd.MultiIndex.from_tuples(
                    [
                        (category, *downstream)
                        for downstream in pd.MultiIndex.from_product([[feature_column], tukey_df.index])
                    ],
                    names=("Category", "Feature", "Group"),
                )
                result.append(tukey_df)
        pd.concat(result).to_excel(self.analysis_path / "tukey.xlsx")

    def categorical_to_feature_pairwise_bonferroni(self):
        for category in self.category_columns:
            for feature_column in self.feature_columns:
                tukey = posthoc_ttest(
                    endog=self.analysis_df.loc[:, feature_column],  # Data
                    groups=self.metadata_df.loc[:, category],  # Groups
                    alpha=0.05,  # Significance
                )
                tukey_df = pd.DataFrame(
                    data=tukey._results_table.data[1:],
                    columns=tukey._results_table.data[0],
                )
                tukey_df.index = pd.MultiIndex.from_tuples(
                    [
                        (category, *downstream)
                        for downstream in pd.MultiIndex.from_product([[feature_column], tukey_df.index])
                    ]
                )
                result.append(tukey_df)
        return pd.concat(result)

    @cached_property
    def merged_df(self):
        pd.concat((self.analysis_df.sort_index(), self.metadata_df.sort_index()), axis=1)

    @cached_property
    def unique_category_values(self):
        result = {}
        for column in self.category_columns:
            assert column in self.metadata_df.columns
            result[column] = np.unique(self.metadata_df[column])
        return result

    @cached_property
    def categorized_dataframes(self) -> dict:
        result = {}
        for column in self.category_columns:
            dataframe_set = []
            for unique_category in self.unique_category_values[column]:
                boolean_index = self.metadata_df[column] == unique_category
                assert np.any(boolean_index)
                df = self.analysis_df[boolean_index]
                df.index = pd.MultiIndex.from_product([[unique_category], df.index], names=("Group", "Animal_ID"))
                dataframe_set.append(df)
            result[column] = pd.concat(dataframe_set)
        return result

    @property
    def analysis_path(self):
        os.makedirs(
            (result := self.project_directory / f"{self.identifier}_statistics"),
            exist_ok=True,
        )
        return result

    @property
    def figure_path(self):
        os.makedirs((result := self.analysis_path / "figures"), exist_ok=True)
        return result
