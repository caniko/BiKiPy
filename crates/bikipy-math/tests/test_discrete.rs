use polars::prelude::*;

use bikipy_math::discrete::*;

#[test]
fn count_true_sums_booleans() {
    let df = df! {
        "flag" => &[true, false, true, true, false],
    }
    .unwrap();

    let result = df
        .lazy()
        .select([count_true_expr("flag").alias("total")])
        .collect()
        .unwrap();

    let total = result.column("total").unwrap().u32().unwrap();
    assert_eq!(total.get(0).unwrap(), 3);
}

#[test]
fn count_true_all_false() {
    let df = df! {
        "flag" => &[false, false, false],
    }
    .unwrap();

    let result = df
        .lazy()
        .select([count_true_expr("flag").alias("total")])
        .collect()
        .unwrap();

    let total = result.column("total").unwrap().u32().unwrap();
    assert_eq!(total.get(0).unwrap(), 0);
}

#[test]
fn any_of_logical_or() {
    let df = df! {
        "a" => &[true, false, false],
        "b" => &[false, true, false],
    }
    .unwrap();

    let result = df
        .lazy()
        .with_column(any_of(vec![col("a"), col("b")]).alias("any"))
        .collect()
        .unwrap();

    let any_col = result.column("any").unwrap().bool().unwrap();
    assert_eq!(any_col.get(0).unwrap(), true);
    assert_eq!(any_col.get(1).unwrap(), true);
    assert_eq!(any_col.get(2).unwrap(), false);
}

#[test]
fn all_of_logical_and() {
    let df = df! {
        "a" => &[true, true, false],
        "b" => &[true, false, true],
    }
    .unwrap();

    let result = df
        .lazy()
        .with_column(all_of(vec![col("a"), col("b")]).alias("all"))
        .collect()
        .unwrap();

    let all_col = result.column("all").unwrap().bool().unwrap();
    assert_eq!(all_col.get(0).unwrap(), true);
    assert_eq!(all_col.get(1).unwrap(), false);
    assert_eq!(all_col.get(2).unwrap(), false);
}
