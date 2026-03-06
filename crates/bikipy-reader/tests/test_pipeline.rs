use polars::prelude::*;

use bikipy_reader::pipeline::{AugmentedBuilder, PipelineStep};

struct AddOneStep;

impl PipelineStep for AddOneStep {
    fn name(&self) -> &str {
        "add_one"
    }

    fn apply(&self, lf: LazyFrame) -> LazyFrame {
        lf.with_column((col("x") + lit(1.0)).alias("x"))
    }
}

struct DoubleStep;

impl PipelineStep for DoubleStep {
    fn name(&self) -> &str {
        "double"
    }

    fn apply(&self, lf: LazyFrame) -> LazyFrame {
        lf.with_column((col("x") * lit(2.0)).alias("x"))
    }
}

#[test]
fn builder_empty_passthrough() {
    let builder = AugmentedBuilder::new();
    let df = df! { "x" => &[1.0, 2.0, 3.0] }.unwrap();
    let result = builder.build_and_collect(df.lazy()).unwrap();
    let x = result.column("x").unwrap().f64().unwrap();
    assert!((x.get(0).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn builder_single_step() {
    let builder = AugmentedBuilder::new().add_step(AddOneStep);
    let df = df! { "x" => &[1.0, 2.0] }.unwrap();
    let result = builder.build_and_collect(df.lazy()).unwrap();
    let x = result.column("x").unwrap().f64().unwrap();
    assert!((x.get(0).unwrap() - 2.0).abs() < 1e-10);
    assert!((x.get(1).unwrap() - 3.0).abs() < 1e-10);
}

#[test]
fn builder_chained_steps() {
    let builder = AugmentedBuilder::new()
        .add_step(AddOneStep)
        .add_step(DoubleStep);
    let df = df! { "x" => &[1.0] }.unwrap();
    let result = builder.build_and_collect(df.lazy()).unwrap();
    let x = result.column("x").unwrap().f64().unwrap();
    // (1 + 1) * 2 = 4
    assert!((x.get(0).unwrap() - 4.0).abs() < 1e-10);
}

#[test]
fn builder_lazy_returns_lazyframe() {
    let builder = AugmentedBuilder::new().add_step(AddOneStep);
    let df = df! { "x" => &[5.0] }.unwrap();
    let lf = builder.build(df.lazy());
    let result = lf.collect().unwrap();
    let x = result.column("x").unwrap().f64().unwrap();
    assert!((x.get(0).unwrap() - 6.0).abs() < 1e-10);
}

#[test]
fn builder_sink_parquet() {
    let builder = AugmentedBuilder::new().add_step(AddOneStep);
    let df = df! { "x" => &[10.0, 20.0] }.unwrap();

    let tmp = std::env::temp_dir().join("bikipy_test_sink.parquet");
    builder.build_and_sink_parquet(df.lazy(), &tmp).unwrap();

    // Read back
    assert!(tmp.exists());
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn builder_default() {
    let _builder = AugmentedBuilder::default();
}
