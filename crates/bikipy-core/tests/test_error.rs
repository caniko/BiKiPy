use bikipy_core::error::*;

#[test]
fn error_display_config() {
    let err = BikipyError::Config("bad setting".to_string());
    assert_eq!(format!("{err}"), "Configuration error: bad setting");
}

#[test]
fn error_display_data() {
    let err = BikipyError::Data("corrupt file".to_string());
    assert_eq!(format!("{err}"), "Data error: corrupt file");
}

#[test]
fn error_display_shape() {
    let err = BikipyError::Shape("invalid polygon".to_string());
    assert_eq!(format!("{err}"), "Shape error: invalid polygon");
}

#[test]
fn error_display_missing_column() {
    let err = BikipyError::MissingColumn("nose_x".to_string());
    assert_eq!(format!("{err}"), "Missing column: nose_x");
}

#[test]
fn error_display_other() {
    let err = BikipyError::Other("unexpected".to_string());
    assert_eq!(format!("{err}"), "unexpected");
}

#[test]
fn error_from_io() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "not found");
    let err: BikipyError = io_err.into();
    assert!(format!("{err}").contains("not found"));
}

#[test]
fn result_type_ok() {
    let r: Result<i32> = Ok(42);
    assert!(matches!(r, Ok(42)));
}

#[test]
fn result_type_err() {
    let r: Result<i32> = Err(BikipyError::Data("test".into()));
    assert!(r.is_err());
}
