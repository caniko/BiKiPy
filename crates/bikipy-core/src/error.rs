use thiserror::Error;

#[derive(Error, Debug)]
pub enum BikipyError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Polars error: {0}")]
    Polars(#[from] polars::prelude::PolarsError),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Data error: {0}")]
    Data(String),

    #[error("Shape error: {0}")]
    Shape(String),

    #[error("Missing column: {0}")]
    MissingColumn(String),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, BikipyError>;
